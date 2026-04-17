import argparse
import json
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

#新增
def resolve_overlay_wsi_path(row, wsi_dir=None):
    # 1) 优先从 splits_csv 的 wsi_path 直接取
    if "wsi_path" in row and pd.notna(row["wsi_path"]):
        p = Path(str(row["wsi_path"]))
        if p.exists():
            return p

    # 2) 再从传入的根目录递归搜索
    if wsi_dir is not None:
        slide_id = str(row["slide_id"])
        root = Path(wsi_dir)
        cands = []
        for ext in ("*.tif", "*.tiff", "*.svs"):
            cands.extend(root.rglob(f"{slide_id}{ext[1:]}"))
        if len(cands) > 0:
            return sorted(cands)[0]

    return None
# --- 优雅降级：可选依赖项 ---
try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    print(" Warning: OpenSlide not installed, overlay heatmaps will be skipped.")

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print(" Warning: SciPy not installed, heatmap smoothing will be disabled.")

# --- 项目内依赖 ---
from wsi_mil.utils.io import read_json
from wsi_mil.models.wsi_mil_model import WSIBaselineMIL
from wsi_mil.utils.vis import save_topk_mosaic
from wsi_mil.datasets.bag_dataset import build_transforms

def load_attention_weights(attention_dir, slide_id):
    """加载08脚本导出的attention权重"""
    if not attention_dir:
        return None
    attn_file = Path(attention_dir) / f"{slide_id}_attn.pt"
    if attn_file.exists():
        return torch.load(attn_file, map_location="cpu")
    return None

class SlideTileDataset(torch.utils.data.Dataset):
    def __init__(self, tile_records, transform):
        self.tile_records = tile_records
        self.transform = transform
        
    def __len__(self):
        return len(self.tile_records)
        
    def __getitem__(self, idx):
        path = self.tile_records[idx]["tile_path"]
        img = Image.open(path).convert("RGB")
        return self.transform(img)

@torch.no_grad()
def infer_slide_all_tiles(model, tile_records, device, img_size: int, tile_bs: int = 64, num_workers: int = 4):
    model.eval()
    torch.cuda.empty_cache()
    
    transform = build_transforms(train=False, img_size=img_size)
    
    tile_ds = SlideTileDataset(tile_records, transform)
    tile_loader = DataLoader(tile_ds, batch_size=tile_bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    z_list = []
    
    for X in tile_loader:
        X = X.to(device, non_blocking=True)
        with autocast( enabled=True):
            zb = model.encoder(X)
        z_list.append(zb.cpu())
    
    z = torch.cat(z_list, dim=0).unsqueeze(0).to(device)
    
    with autocast( enabled=True):
        slide_logit, alpha, h, e = model.mil(z)
        slide_prob = torch.sigmoid(slide_logit).item()
        
    alpha = alpha.view(-1).cpu().numpy() 
    e = e.squeeze(0).cpu().numpy()
    
    torch.cuda.empty_cache()
    return slide_prob, alpha, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/stage1_baseline.yaml")
    ap.add_argument("--ckpt", type=str, default="runs/stage1_baseline/ckpt_best.pt")
    ap.add_argument("--bag_index", type=str, default="data/metadata/bag_index.json")
    ap.add_argument("--splits_csv", type=str, default="data/metadata/splits_patient.csv")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--tile_bs", type=int, default=64)
    ap.add_argument("--wsi_dir", type=str, default=None, help="原始WSI目录，用于绘制Overlay热力图") 
    ap.add_argument("--attn_dir", type=str, default=None, help="如果提供，则直接读取08导出的Attention，跳过模型推理") 
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(cfg["experiment"]["out_dir"])
    topk_dir = out_dir / "topk_tiles"
    heat_grid_dir = out_dir / "heatmaps_grid"
    heat_overlay_dir = out_dir / "heatmaps_overlay"
    
    topk_dir.mkdir(parents=True, exist_ok=True)
    heat_grid_dir.mkdir(parents=True, exist_ok=True)
    heat_overlay_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = WSIBaselineMIL(**cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)

    bags = read_json(args.bag_index)
    df = pd.read_csv(args.splits_csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    
    tile_size = cfg["data"].get("tile_size", 256) 
    num_workers = cfg["data"].get("num_workers", 4)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Exporting Visualization"):
        slide_id = str(r["slide_id"])
        if slide_id not in bags:
            continue
            
        tile_records = bags[slide_id]
        if len(tile_records) == 0:
            continue
            
        # --- 核心改进：Attention 缓存与降级推理 ---
        alpha = None
        slide_prob = None
        
        # 1. 尝试从 08 的产出中直接读取
        if args.attn_dir:
            saved_data = load_attention_weights(args.attn_dir, slide_id)
            if saved_data:
                alpha = saved_data["attention"].numpy()
                slide_prob = saved_data["prob"]
                print(f"✅ 使用缓存的 Attention: {slide_id}")
        
        # 2. 降级：如果没有提供目录，或者文件不存在（比如这是张新切片），则启动重新推理
        if alpha is None:
            print(f"⚡ 重新推理 Attention: {slide_id}")
            slide_prob, alpha, _e = infer_slide_all_tiles(
                model, tile_records, device, cfg["data"]["img_size"], args.tile_bs, num_workers
            )

        # 1. 导出 Top-K Mosaic
        idx = np.argsort(-alpha)[: args.topk]
        top_paths = [tile_records[i]["tile_path"] for i in idx]
        top_scores = [float(alpha[i]) for i in idx]
        mosaic_png = topk_dir / f"{slide_id}.png"
        save_topk_mosaic(top_paths, top_scores, str(mosaic_png), thumb_size=cfg["data"]["img_size"])

        xs = np.array([int(tr["x"]) for tr in tile_records], dtype=np.int32)
        ys = np.array([int(tr["y"]) for tr in tile_records], dtype=np.int32)
        
        # 2. 绘制版本一：纯 Attention Grid 热力图
        if len(xs) > 0:
            x0, y0 = xs.min(), ys.min()
            x1, y1 = xs.max(), ys.max()

            w = int((x1 - x0) / tile_size) + 1
            h = int((y1 - y0) / tile_size) + 1

            heat = np.zeros((h, w), dtype=np.float32)
            count = np.zeros((h, w), dtype=np.float32)

            for tr, a in zip(tile_records, alpha):
                gx = int((int(tr["x"]) - x0) / tile_size)
                gy = int((int(tr["y"]) - y0) / tile_size)
                heat[gy, gx] += float(a)
                count[gy, gx] += 1.0

            heat = heat / np.maximum(count, 1e-6)
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            
            # 对纯 Grid 也应用平滑（提升视觉效果）
            if HAS_SCIPY:
                heat = gaussian_filter(heat, sigma=1)

            grid_png = heat_grid_dir / f"{slide_id}.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(heat, cmap="jet")
            plt.colorbar()
            plt.title(f"{slide_id} | Prob={slide_prob:.3f}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(grid_png, dpi=150)
            plt.close()

        # 3. 绘制版本二：叠加到 WSI 缩略图上的 Overlay 热力图
        if args.wsi_dir and HAS_OPENSLIDE:
            # 优先用 splits_csv 中的 wsi_path，找不到再去你传的正式数据根目录递归查找
            wsi_path = resolve_overlay_wsi_path(r, args.wsi_dir)
            
            slide = None
            fig = None
            try:
                if wsi_path:
                    slide = openslide.OpenSlide(str(wsi_path))
                    thumb = slide.get_thumbnail((2048, 2048))
                    t_w, t_h = thumb.size
                    
                    orig_w, orig_h = slide.dimensions
                    scale_x, scale_y = t_w / orig_w, t_h / orig_h
                    
                    full_heat = np.zeros((t_h, t_w), dtype=np.float32)
                    
                    for x_val, y_val, a in zip(xs, ys, alpha):
                        cx = int(x_val * scale_x)
                        cy = int(y_val * scale_y)
                        
                        patch_w = max(1, int(tile_size * scale_x))
                        patch_h = max(1, int(tile_size * scale_y))
                        
                        # 修复边界检查：防止越界
                        ex = min(cx + patch_w, t_w)
                        ey = min(cy + patch_h, t_h)
                        if cx < t_w and cy < t_h:
                            full_heat[cy:ey, cx:ex] = float(a)
                    
                    # 归一化并进行轻微高斯平滑 (由 HAS_SCIPY 保护)
                    full_heat = (full_heat - full_heat.min()) / (full_heat.max() - full_heat.min() + 1e-8)
                    if HAS_SCIPY:
                        full_heat = gaussian_filter(full_heat, sigma=2)
                    
                    overlay_png = heat_overlay_dir / f"{slide_id}_overlay.png"
                    fig = plt.figure(figsize=(10, 10))
                    plt.imshow(thumb)
                    plt.imshow(full_heat, cmap="jet", alpha=0.5) 
                    plt.axis("off")
                    plt.title(f"{slide_id} | Overlay Heatmap | Prob={slide_prob:.3f}")
                    plt.tight_layout()
                    plt.savefig(overlay_png, dpi=200)
                    plt.close(fig)
            except Exception as e:
                print(f"⚠️ 处理 {slide_id} Overlay 失败: {e}")
            finally:
                if slide:
                    slide.close()

        # 4. 记录数据
        for rank, i in enumerate(idx, start=1):
            rows.append(
                dict(
                    slide_id=slide_id,
                    rank=rank,
                    tile_path=tile_records[i]["tile_path"],
                    alpha=float(alpha[i]),
                    x=int(tile_records[i]["x"]),
                    y=int(tile_records[i]["y"]),
                    slide_prob=float(slide_prob),
                    y_true=int(r["label"]),
                )
            )
        
        # 内存管理：清理当前slide的缓存
        gc.collect()
        torch.cuda.empty_cache()

    out_csv = out_dir / "topk_tiles.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("导出完毕, Top-K 清单已保存至:", out_csv)

if __name__ == "__main__":
    main()