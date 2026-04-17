import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def has_both_classes(df: pd.DataFrame) -> bool:
    s = set(df["label"].astype(int).unique().tolist())
    return s == {0, 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides_csv", type=str, default="data/metadata/slides_raw.csv")
    ap.add_argument("--out_splits", type=str, default="data/metadata/splits_patient.csv")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_retry", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.slides_csv)

    required = {"slide_id", "label", "patient_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必需列: {missing}")

    df["slide_id"] = df["slide_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df["label"] = df["label"].astype(int)

    val_rel = args.val_ratio / (1.0 - args.train_ratio)

    best = None
    for retry in range(args.max_retry):
        seed = args.seed + retry

        gss1 = GroupShuffleSplit(n_splits=1, train_size=args.train_ratio, random_state=seed)
        train_idx, temp_idx = next(gss1.split(df, groups=df["patient_id"]))
        df_train = df.iloc[train_idx].copy()
        df_temp = df.iloc[temp_idx].copy()

        gss2 = GroupShuffleSplit(n_splits=1, train_size=val_rel, random_state=seed)
        val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["patient_id"]))
        df_val = df_temp.iloc[val_idx].copy()
        df_test = df_temp.iloc[test_idx].copy()

        ok = has_both_classes(df_val) and has_both_classes(df_test)
        if ok:
            best = (seed, df_train, df_val, df_test)
            break

    if best is None:
        raise RuntimeError("多次重试后，val/test 仍无法同时包含 0/1 两类，请检查 patient 分布。")

    used_seed, df_train, df_val, df_test = best

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    out = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)
    out = out.sort_values(["split", "label", "patient_id", "slide_id"]).reset_index(drop=True)

    out_path = Path(args.out_splits)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("Used seed:", used_seed)
    print("Split sizes:", out["split"].value_counts().to_dict())
    print("Split x label:")
    print(out.groupby(["split", "label"]).size())
    print("Unique patients by split:")
    print(out.groupby("split")["patient_id"].nunique())
    print("Saved:", out_path)


if __name__ == "__main__":
    main()