import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler 
from tqdm import tqdm

from wsi_mil.utils.metrics import compute_metrics
from wsi_mil.utils.io import mkdir

from PIL import Image
from wsi_mil.datasets.bag_dataset import build_transforms
from wsi_mil.models.wsi_mil_model import ForwardOut

def save_ckpt(path: str, model, optimizer, epoch: int, best_metric: float):
    p = Path(path)
    mkdir(p.parent)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        p,
    )




###stage2 添加辅助函数，实现 lazy full-bag 分支
@torch.no_grad()
def _forward_fullbag_lazy(model, tile_records, device, img_size: int, tile_bs: int = 64, amp: bool = True):
    """
    lazy_load=True 时，bag_imgs 为空，需要从 tile_records 实时加载图块
    """
    transform = build_transforms(train=False, img_size=img_size)
    feats = []

    for i in range(0, len(tile_records), tile_bs):
        xs = []
        for tr in tile_records[i:i + tile_bs]:
            img = Image.open(tr["tile_path"]).convert("RGB")
            xs.append(transform(img))
        xb = torch.stack(xs, dim=0).to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp):
            zb = model.encoder(xb)
        feats.append(zb.cpu())

    z = torch.cat(feats, dim=0).unsqueeze(0).to(device)

    with autocast(device_type="cuda", enabled=amp):
        slide_logit, alpha, h, _ = model.mil(z)
        slide_prob = torch.sigmoid(slide_logit)

    return ForwardOut(
        slide_logit=slide_logit,
        slide_prob=slide_prob,
        alpha=alpha,
        h=h,
        tile_z=z,
        tile_logit=None,
        tile_prob=None,
        tile_u=None,
    )


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = True, tile_bs: int = 128, img_size: int = 224) -> Dict[str, Any]:
    """
    tile_bs: chunk size for encoding tiles when bag is large (to avoid OOM)
    img_size: 图块尺寸，用于 lazy_load 模式下的实时加载
    """
    model.eval()
    y_true, y_prob, slide_ids = [], [], []
    losses = []

    for bag_imgs, label, meta in tqdm(loader, desc="eval", leave=False):
        label = label.to(device)

        with autocast(device_type="cuda", enabled=amp):
            # stage2: lazy full-bag 模式，bag_imgs 是空占位 tensor，从 tile_records 加载
            if bag_imgs.numel() == 0:
                tile_records = meta["tile_records"][0] if isinstance(meta["tile_records"], list) else meta["tile_records"]
                out = _forward_fullbag_lazy(
                    model=model,
                    tile_records=tile_records,
                    device=device,
                    img_size=img_size,
                    tile_bs=tile_bs,
                    amp=amp,
                )
            else:
                # 原代码逻辑：非 lazy 模式，bag_imgs 包含真实图像数据
                bag_imgs = bag_imgs.to(device, non_blocking=True)
                B, N = bag_imgs.shape[:2]

                if N <= tile_bs:
                    out = model(bag_imgs)
                else:
                    # Chunked encoding for large bags
                    # Encode tiles in chunks to avoid OOM
                    z_list = []
                    for i in range(0, N, tile_bs):
                        chunk = bag_imgs[:, i:i+tile_bs].reshape(-1, *bag_imgs.shape[2:])  # [chunk_size, 3, H, W]
                        z_chunk = model.encoder(chunk)  # [chunk_size, D]
                        z_list.append(z_chunk)
                    z = torch.cat(z_list, dim=0).unsqueeze(0)  # [1, N, D]
                    # MIL aggregation
                    slide_logit, alpha, h, _ = model.mil(z)
                    slide_prob = torch.sigmoid(slide_logit)
                    #兼容stage1、stage2
                    out = ForwardOut(
                        slide_logit=slide_logit,
                        slide_prob=slide_prob,
                        alpha=alpha,
                        h=h,
                        tile_z=z,
                        tile_logit=None,
                        tile_prob=None,
                        tile_u=None,
                    )

            logit = out.slide_logit
            loss = F.binary_cross_entropy_with_logits(logit, label.float())

        losses.append(loss.item())

        prob = out.slide_prob.detach().cpu().numpy().tolist()
        y_prob += prob
        y_true += label.detach().cpu().numpy().tolist()
        slide_ids += meta["slide_id"] if isinstance(meta["slide_id"], list) else [meta["slide_id"]]

    m = compute_metrics(y_true, y_prob, thr=0.5)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auc": m.auc,
        "f1": m.f1,
        "sensitivity": m.sensitivity,
        "specificity": m.specificity,
        "y_true": y_true,
        "y_prob": y_prob,
        "slide_id": slide_ids,
    }

def train_one_epoch(model, loader, optimizer, device, scaler: GradScaler, amp: bool = True) -> float:
    model.train()
    losses = []

    for bag_imgs, label, _meta in tqdm(loader, desc="train", leave=False):
        bag_imgs = bag_imgs.to(device, non_blocking=True)
        label = label.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=amp):
            out = model(bag_imgs)
            logit = out.slide_logit
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")