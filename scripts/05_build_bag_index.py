import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_manifest", type=str, default="data/metadata/tile_manifest.csv")
    ap.add_argument("--out_json", type=str, default="data/metadata/bag_index.json")
    ap.add_argument("--out_stats_csv", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.tile_manifest)
    if df.empty:
        raise RuntimeError("tile_manifest.csv 为空")

    df = df.sort_values(["slide_id", "y", "x", "tile_path"]).reset_index(drop=True)

    bags = defaultdict(list)
    for _, r in df.iterrows():
        bags[str(r["slide_id"])].append(
            dict(
                tile_path=str(r["tile_path"]),
                x=int(r["x"]),
                y=int(r["y"]),
                tissue_ratio=float(r["tissue_ratio"]),
            )
        )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bags, f, ensure_ascii=False)

    stats = (
        df.groupby("slide_id")
          .agg(
              n_tiles=("tile_path", "count"),
              mean_tissue_ratio=("tissue_ratio", "mean"),
          )
          .reset_index()
          .sort_values("slide_id")
    )

    out_stats_csv = Path(args.out_stats_csv) if args.out_stats_csv else out.with_name("bag_index_stats.csv")
    stats.to_csv(out_stats_csv, index=False, encoding="utf-8-sig")

    print("Saved:", out, "n_slides=", len(bags))
    print("Saved:", out_stats_csv)


if __name__ == "__main__":
    main()