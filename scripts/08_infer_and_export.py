import argparse
import json
import yaml
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from wsi_mil.datasets.bag_dataset import SlideBagDataset, build_transforms
from wsi_mil.models.wsi_mil_model import WSIBaselineMIL
from wsi_mil.utils.seed import seed_everything
from wsi_mil.utils.log import get_logger
from wsi_mil.utils.metrics import compute_metrics

class SlideTileDataset(torch.utils.data.Dataset):
    """用于推理期多进程提图的轻量级 Dataset"""
    def __init__(self, tile_records, transform):
        self.tile_records = tile_records
        self.transform = transform
        
    def __len__(self):
        return len(self.tile_records)
        
    def __getitem__(self, idx):
        from PIL import Image
        path = self.tile_records[idx]["tile_path"]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.tile_records[idx]  # 顺便返回 meta 信息用于记录坐标

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/stage1_baseline.yaml")
    ap.add_argument("--ckpt", type=str, required=True, help="模型权重路径, 例如 ckpt_best.pt")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="允许对任意 split 跑全图推理")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--eval_tile_bs", type=int, default=64, help="Full-bag 提特征分块大小")
    ap.add_argument("--out_dir", type=str, default="results/inference", help="预测结果和 Attention 权重的输出目录")
    ap.add_argument("--save_attention", action="store_true", help="是否导出每个图块的 Attention 分数用于画热力图")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(args.out_dir) / args.split
    attention_dir = out_dir / "attention_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_attention:
        attention_dir.mkdir(parents=True, exist_ok=True)
        
    logger = get_logger(str(out_dir / f"infer_{args.split}.log"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["experiment"]["seed"], deterministic=False)

    logger.info(f"初始化 Full-Bag 推理 | Split: {args.split} | Weights: {args.ckpt}")

    # 1. 严格使用 full_bag=True 和 lazy_load=True
    dataset = SlideBagDataset(
        splits_csv=cfg["data"]["splits_csv"],
        bag_index_json=cfg["data"]["bag_index_json"],
        split=args.split, 
        bag_size=cfg["data"]["bag_size"],
        img_size=cfg["data"]["img_size"], 
        seed=cfg["experiment"]["seed"],
        full_bag=True, 
        lazy_load=True
    )
    
    transform = build_transforms(train=False, img_size=cfg["data"]["img_size"])

    # 2. 加载模型
    model = WSIBaselineMIL(**cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    logger.info("模型权重加载完成")

    results_list = []
    
    # 3. 开始遍历切片 (加入 tqdm 进度条)
    for i in tqdm(range(len(dataset)), desc=f"Inferring {args.split}"):
        _, label, meta = dataset[i]
        slide_id = meta["slide_id"]
        label_val = label.item()
        tile_records = meta["tile_records"]
        n_tiles = len(tile_records)
        
        if n_tiles == 0:
            logger.warning(f"Slide {slide_id} 为空，跳过。")
            continue

        # 多进程 Dataloader 防止 I/O 瓶颈
        tile_ds = SlideTileDataset(tile_records, transform)
        tile_loader = DataLoader(
            tile_ds, batch_size=args.eval_tile_bs, 
            shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True
        )
        
        features_list = []
        coords_list = []  # 记录坐标用于后续 Attention 热力图对齐
        
        # 分块提特征
        for X, batch_meta in tile_loader:
            X = X.to(device, non_blocking=True)
            with autocast( enabled=cfg["train"]["amp"]):
                chunk_feat = model.encoder(X) 
            features_list.append(chunk_feat.cpu())
            
            if args.save_attention:
                # 修复隐患1：直接利用 collate 好的 Tensor
                # 确保转为数值型，防止上游数据类型是字符串
                cy = batch_meta['y'].clone().detach().cpu()
                cx = batch_meta['x'].clone().detach().cpu()
                coords = torch.stack([cy, cx], dim=1)
                coords_list.append(coords)
            
        # 拼接整图特征过 MIL 头
        full_features = torch.cat(features_list, dim=0).unsqueeze(0).to(device)
        
        with autocast( enabled=cfg["train"]["amp"]):
            outputs = model.mil(full_features)
            slide_logit = outputs[0]
            attention_weights = outputs[1] if len(outputs) > 1 else None 
            
            slide_prob = torch.sigmoid(slide_logit).item()
            pred_class = 1 if slide_prob >= 0.5 else 0 

        del full_features
        torch.cuda.empty_cache()
        
        # --- 4. 追加审计字段与预测结果 ---
        results_list.append({
            "slide_id": slide_id,
            "y_true": label_val,
            "y_prob": slide_prob,
            "y_pred": pred_class,
            "n_tiles": n_tiles,
            "used_full_bag": True,
            "tile_bs": args.eval_tile_bs
        })
        
        # --- 5. 导出 Attention 用于后续 09 脚本可视化 ---
        if args.save_attention and attention_weights is not None:
            all_coords = torch.cat(coords_list, dim=0)
            
            # 修复隐患2：用 view(-1) 替代 squeeze() 防止 N=1 时变成标量报错
            att_scores = attention_weights.view(-1).cpu()
            
            torch.save({
                "slide_id": slide_id,
                "coords": all_coords,
                "attention": att_scores,
                "prob": slide_prob
            }, attention_dir / f"{slide_id}_attn.pt")

    # 6. 保存最终的 CSV 报告
    df_res = pd.DataFrame(results_list)
    csv_path = out_dir / f"{args.split}_predictions.csv"
    df_res.to_csv(csv_path, index=False)
    
    # 计算指标
    m = compute_metrics(df_res["y_true"].tolist(), df_res["y_prob"].tolist(), thr=0.5)
    metrics = {"loss": float("nan"), "auc": m.auc, "f1": m.f1, 
               "sensitivity": m.sensitivity, "specificity": m.specificity}
    
    # 审计信息
    audit_info = {
        "total_slides": len(df_res),
        "avg_tiles_per_slide": float(df_res["n_tiles"].mean()) if len(df_res) > 0 else 0,
        "total_tiles": int(df_res["n_tiles"].sum()) if len(df_res) > 0 else 0,
        "used_full_bag": True,
        "tile_bs": args.eval_tile_bs,
        "split": args.split,
        "saved_attention": args.save_attention
    }
    metrics.update({"audit": audit_info})
    
    # 保存 metrics
    metric_path = out_dir / f"metrics_{args.split}.json"
    metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    logger.info(f" 推理完成！Split: {args.split} | AUC: {metrics['auc']:.4f} | ACC: {m.f1:.4f}")
    logger.info(f" 审计与预测报告已保存至: {csv_path}")
    if args.save_attention:
        logger.info(f"Attention 权重已保存至: {attention_dir}")

if __name__ == "__main__":
    main()