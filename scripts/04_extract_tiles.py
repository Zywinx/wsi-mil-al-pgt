import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from wsi_mil.utils.wsi_reader import open_wsi


def make_tissue_mask(slide, mask_max_dim=4096):
    w, h = slide.dimensions
    scale = max(w, h) / mask_max_dim if max(w, h) > mask_max_dim else 1.0
    thumb_w, thumb_h = max(1, int(w / scale)), max(1, int(h / scale))
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    arr = np.array(thumb)

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask = ((sat > 20) & (val < 245)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask, scale


def tile_tissue_ratio(mask, scale, x, y, tile_size):
    mx0 = int(x / scale)
    my0 = int(y / scale)
    mx1 = int((x + tile_size) / scale)
    my1 = int((y + tile_size) / scale)

    H, W = mask.shape
    mx0 = max(0, min(mx0, W))
    mx1 = max(0, min(mx1, W))
    my0 = max(0, min(my0, H))
    my1 = max(0, min(my1, H))
    if mx1 <= mx0 or my1 <= my0:
        return 0.0

    region = mask[my0:my1, mx0:mx1]
    return float(region.mean())


def resolve_wsi_path(slide_row, raw_dir: Path | None):
    # 优先使用 splits_csv 中已有的 wsi_path
    if "wsi_path" in slide_row and pd.notna(slide_row["wsi_path"]):
        p = Path(str(slide_row["wsi_path"]))
        if p.exists():
            return p

    if raw_dir is None:
        return None

    slide_id = str(slide_row["slide_id"])
    cands = []
    for ext in ("*.tif", "*.tiff", "*.svs"):
        cands.extend(raw_dir.rglob(f"{slide_id}{ext[1:]}"))
    if len(cands) > 0:
        return sorted(cands)[0]
    return None


def parse_existing_tile_xy(tile_path: Path):
    m = re.search(r"_x(\d+)_y(\d+)\.png$", tile_path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", type=str, default="data/metadata/splits_patient.csv")
    ap.add_argument("--raw_dir", type=str, default=None)
    ap.add_argument("--tiles_dir", type=str, default="data/tiles")
    ap.add_argument("--manifest_csv", type=str, default="data/metadata/tile_manifest.csv")
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--tissue_ratio_thr", type=float, default=0.5)
    ap.add_argument("--mask_max_dim", type=int, default=4096)
    ap.add_argument("--limit_slides", type=int, default=-1)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--reuse_existing_tiles", action="store_true")
    args = ap.parse_args()

    assert 0 <= args.shard_rank < args.num_shards, "shard_rank must be in [0, num_shards)"

    df = pd.read_csv(args.splits_csv)
    if "slide_id" not in df.columns:
        raise ValueError("splits_csv 缺少 slide_id 列")

    slide_df = (
        df.drop_duplicates(subset=["slide_id"])
          .sort_values("slide_id")
          .reset_index(drop=True)
    )

    if args.limit_slides > 0:
        slide_df = slide_df.iloc[: args.limit_slides].copy()

    slide_df = slide_df.iloc[args.shard_rank::args.num_shards].copy().reset_index(drop=True)

    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    tiles_dir = Path(args.tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for _, row in tqdm(slide_df.iterrows(), total=len(slide_df), desc=f"extract shard {args.shard_rank}/{args.num_shards}"):
        slide_id = str(row["slide_id"])
        wsi_path = resolve_wsi_path(row, raw_dir)

        if wsi_path is None or not wsi_path.exists():
            print(f"[WARN] 未找到原始文件: slide_id={slide_id}")
            continue

        out_dir = tiles_dir / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            slide = open_wsi(str(wsi_path))
        except Exception as e:
            print(f"[WARN] 打开失败 {wsi_path}: {e}")
            continue

        try:
            w, h = slide.dimensions
            mask, scale = make_tissue_mask(slide, mask_max_dim=args.mask_max_dim)

            existing = {}
            if args.reuse_existing_tiles and out_dir.exists():
                for p in sorted(out_dir.glob("*.png")):
                    xy = parse_existing_tile_xy(p)
                    if xy is not None:
                        existing[xy] = p

            tile_idx = 0
            for y in range(0, max(1, h - args.tile_size + 1), args.stride):
                for x in range(0, max(1, w - args.tile_size + 1), args.stride):
                    ratio = tile_tissue_ratio(mask, scale, x, y, args.tile_size)
                    if ratio < args.tissue_ratio_thr:
                        continue

                    tile_name = f"{slide_id}_x{x}_y{y}.png"
                    tile_path = out_dir / tile_name

                    if (x, y) in existing and existing[(x, y)].exists():
                        pass
                    else:
                        try:
                            region = slide.read_region((x, y), 0, (args.tile_size, args.tile_size)).convert("RGB")
                            region.save(tile_path, compress_level=1)
                        except Exception as e:
                            print(f"[WARN] read_region/save 失败 slide={slide_id} x={x} y={y}: {e}")
                            continue

                    manifest_rows.append(
                        dict(
                            slide_id=slide_id,
                            tile_id=tile_idx,
                            x=x,
                            y=y,
                            tile_size=args.tile_size,
                            stride=args.stride,
                            tissue_ratio=ratio,
                            tile_path=str(tile_path.resolve()),
                            wsi_path=str(wsi_path.resolve()),
                        )
                    )
                    tile_idx += 1
        finally:
            slide.close()

    manifest_csv = Path(args.manifest_csv)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    print("Saved:", manifest_csv, "n_rows=", len(manifest_rows))


if __name__ == "__main__":
    main()