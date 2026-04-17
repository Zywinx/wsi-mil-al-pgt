import argparse
import json
import yaml
import torch
from pathlib import Path
from torch.amp import autocast
from torch.utils.data import DataLoader

from wsi_mil.datasets.bag_dataset import SlideBagDataset, build_transforms
from wsi_mil.models.wsi_mil_model import WSIBaselineMIL
from wsi_mil.train.trainer import train_one_epoch, evaluate, save_ckpt
from wsi_mil.utils.seed import seed_everything
from wsi_mil.utils.log import get_logger
from wsi_mil.utils.io import mkdir
from wsi_mil.utils.metrics import compute_metrics

class SlideTileDataset(torch.utils.data.Dataset):
    """专门为 Full-bag 推理设计的轻量化内部 Dataset"""
    def __init__(self, tile_records, transform):
        self.tile_records = tile_records
        self.transform = transform
        
    def __len__(self):
        return len(self.tile_records)
        
    def __getitem__(self, idx):
        from PIL import Image
        path = self.tile_records[idx]["tile_path"]
        img = Image.open(path).convert("RGB")
        return self.transform(img)

@torch.no_grad()
def evaluate_full_bag(model, ds_val_full, device, cfg, tile_bs: int = 64):
    """【核心】多进程数据预读 + 分块无损 Full-Bag 验证逻辑"""
    model.eval()
    torch.cuda.empty_cache()
    
    transform = build_transforms(train=False, img_size=cfg["data"]["img_size"])
    
    y_true_list = []
    y_prob_list = []
    
    eval_num_workers = cfg["data"].get("eval_num_workers", cfg["data"].get("num_workers", 2))
    
    for i in range(len(ds_val_full)):
        _, label, meta = ds_val_full[i]
        label_val = label.item()
        tile_records = meta["tile_records"]
        
        if len(tile_records) == 0:
            continue
            
        # --- 核心性能优化：利用 DataLoader 多进程加载图块 ---
        tile_ds = SlideTileDataset(tile_records, transform)
        tile_loader = DataLoader(
            tile_ds,
            batch_size=tile_bs,
            shuffle=False,
            num_workers=eval_num_workers,
            pin_memory=True,
            persistent_workers=(eval_num_workers > 0),
            prefetch_factor=4 if eval_num_workers > 0 else None,
        )
        
        features_list = []
        
        # GPU 此时会源源不断地收到数据，不再会有 I/O 阻塞
        for X in tile_loader:
            X = X.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', enabled=cfg["train"]["amp"]):
                chunk_feat = model.encoder(X) 
                
            features_list.append(chunk_feat.cpu())
            
        # 拼接整图特征，重新放入 GPU
        full_features = torch.cat(features_list, dim=0).unsqueeze(0).to(device)
        
        with autocast(device_type='cuda', enabled=cfg["train"]["amp"]):
            slide_logit, _, _, _ = model.mil(full_features)
            slide_prob = torch.sigmoid(slide_logit).item()
        
        del full_features
        
        y_true_list.append(label_val)
        y_prob_list.append(slide_prob)
    
    m = compute_metrics(y_true_list, y_prob_list, thr=0.5)
    metrics = {
        "loss": float("nan"), 
        "auc": m.auc, 
        "f1": m.f1, 
        "sensitivity": m.sensitivity, 
        "specificity": m.specificity
    }
    
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/stage1_baseline.yaml")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--eval_tile_bs", type=int, default=64, help="Full-bag 分块大小")
    ap.add_argument("--lazy_val_epoch", type=int, default=3, help="前 N 个 Epoch 不跑 Full-bag")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    out_dir = Path(cfg["experiment"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(str(out_dir / "train.log"))

    seed_everything(cfg["experiment"]["seed"], deterministic=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    # --- 数据集初始化（移至外部，避免循环内重复读取 IO） ---
    ds_train = SlideBagDataset(
        splits_csv=cfg["data"]["splits_csv"],
        bag_index_json=cfg["data"]["bag_index_json"],
        split="train", bag_size=cfg["data"]["bag_size"],
        img_size=cfg["data"]["img_size"], seed=cfg["experiment"]["seed"]
    )
    
    ds_val_fixed = SlideBagDataset(
        splits_csv=cfg["data"]["splits_csv"],
        bag_index_json=cfg["data"]["bag_index_json"],
        split="val", bag_size=cfg["data"]["bag_size"],
        img_size=cfg["data"]["img_size"], seed=cfg["experiment"]["seed"]
    )

    # 提前初始化 Full Bag Dataset
    ds_val_full = SlideBagDataset(
        splits_csv=cfg["data"]["splits_csv"],
        bag_index_json=cfg["data"]["bag_index_json"],
        split="val", bag_size=cfg["data"]["bag_size"],
        img_size=cfg["data"]["img_size"], seed=cfg["experiment"]["seed"],
        full_bag=True, lazy_load=True
    )

    def collate_fn(batch):
        bag_imgs, label, meta = zip(*batch)
        bag_imgs = torch.stack(bag_imgs, dim=0)
        label = torch.stack(label, dim=0)
        meta_out = {k: [m[k] for m in meta] for k in meta[0].keys()}
        return bag_imgs, label, meta_out

    train_num_workers = cfg["data"].get("train_num_workers", cfg["data"].get("num_workers", 2))
    eval_num_workers = cfg["data"].get("eval_num_workers", cfg["data"].get("num_workers", 2))

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
        persistent_workers=(train_num_workers > 0),
        prefetch_factor=2 if train_num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    
    val_fixed_loader = DataLoader(
        ds_val_fixed,
        batch_size=1,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=True,
        persistent_workers=(eval_num_workers > 0),
        prefetch_factor=4 if eval_num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    model = WSIBaselineMIL(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg["train"]["amp"])

    log_file = out_dir / "training_metrics.jsonl"
    best_full_auc = 0.0
    
    logger.info(f"🚀 开始训练: {cfg['train']['epochs']} Epochs | Full-bag Lazy: Epoch {args.lazy_val_epoch}")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        logger.info(f"\n--- Epoch {epoch}/{cfg['train']['epochs']} ---")
        
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, amp=cfg["train"]["amp"])
        logger.info(f"[Train] Loss: {tr_loss:.4f}")
        
        val_fixed = evaluate(model, val_fixed_loader, device, amp=cfg["train"]["amp"])
        logger.info(f"[Val - Fixed Bag] Loss: {val_fixed['loss']:.4f} | AUC: {val_fixed['auc']:.4f} | F1: {val_fixed['f1']:.4f}")
        
        val_full_metrics = {}
        if epoch >= args.lazy_val_epoch:
            logger.info("[Val - Full Bag] 正在进行整图无损推理，请稍候...")
            val_full_metrics = evaluate_full_bag(model, ds_val_full, device, cfg, args.eval_tile_bs)
            full_auc = val_full_metrics['auc']
            logger.info(f"[Val - Full Bag] Real Full-Bag AUC: {full_auc:.4f} | F1: {val_full_metrics['f1']:.4f}")
            
            # 仅在突破最佳记录时，保存模型和独立的指标 JSON
            if full_auc > best_full_auc:
                best_full_auc = full_auc
                save_ckpt(str(out_dir / "ckpt_best.pt"), model, optimizer, epoch, best_full_auc)
                
                (out_dir / "best_metrics_val_fixedbag.json").write_text(json.dumps(val_fixed, indent=2), encoding="utf-8")
                (out_dir / "best_metrics_val_fullbag.json").write_text(json.dumps(val_full_metrics, indent=2), encoding="utf-8")
                
                logger.info(f"🎉 发现更好模型，已保存最佳 Full-Bag AUC: {best_full_auc:.4f}")
        else:
            logger.info(f"[Val - Full Bag] 懒加载触发 (Epoch {epoch} < {args.lazy_val_epoch})，跳过整图验证。")
            val_full_metrics = {"auc": None, "f1": None, "sensitivity": None, "specificity": None}
        
        save_ckpt(str(out_dir / "ckpt_last.pt"), model, optimizer, epoch, best_full_auc)
            
        # 所有 Epoch 的流水账数据，集中存入 jsonl 即可
        log_entry = {
            "epoch": epoch,
            "train": {"loss": tr_loss},
            "val_fixed": val_fixed,
            "val_full": val_full_metrics
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    logger.info(f"\n🏁 训练结束！最佳 Full-Bag AUC: {best_full_auc:.4f}")

if __name__ == "__main__":
    main()