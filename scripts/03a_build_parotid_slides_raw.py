#扫描 Benign/ 和 Malignant/；生成 slides_raw.csv；
import argparse
import re
from pathlib import Path
import pandas as pd


LABEL_MAP = {
    "Benign": 0,
    "Malignant": 1,
}


def extract_patient_id(stem: str) -> str:
    """
    从文件名主干提取 patient_id。
    规则：
    - 优先匹配 F23-00306 / F21-06724 / F22-00471 这种
    - 其次匹配 22-36133 / 21-12345 这种
    - patient_id 只保留病理号主干，不带 A01 / H01 / 重切 / 医生名
    """
    s = stem.strip()

    m = re.search(r"(F\d{2}-\d{5})", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"(\d{2}-\d{5})", s)
    if m:
        return m.group(1)

    raise ValueError(f"无法从文件名提取 patient_id: {stem}")


def scan_one_dir(class_dir: Path, label_text: str):
    rows = []
    tif_files = sorted(list(class_dir.glob("*.tif")) + list(class_dir.glob("*.tiff")))
    label = LABEL_MAP[label_text]

    for p in tif_files:
        stem = p.stem.strip()
        patient_id = extract_patient_id(stem)

        rows.append(
            {
                "slide_id": stem,             # 工程里 slide 唯一键
                "file_id": stem,
                "filename": p.name,
                "wsi_path": str(p.resolve()),
                "label": label,               # benign=0, malignant=1
                "label_text": label_text,
                "patient_id": patient_id,
                "ext": p.suffix.lower(),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help='正式数据根目录，例如 "/data/maxinyu/WSI WORKSPACE/data/Parotid/sdpc to tif"',
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="data/metadata/parotid_slides_raw.csv",
    )
    args = ap.parse_args()

    root = Path(args.root_dir)
    benign_dir = root / "Benign"
    malignant_dir = root / "Malignant"

    if not benign_dir.exists():
        raise FileNotFoundError(f"未找到目录: {benign_dir}")
    if not malignant_dir.exists():
        raise FileNotFoundError(f"未找到目录: {malignant_dir}")

    rows = []
    rows += scan_one_dir(benign_dir, "Benign")
    rows += scan_one_dir(malignant_dir, "Malignant")

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("没有扫描到任何 tif/tiff 文件")

    if df["slide_id"].duplicated().any():
        dup = df[df["slide_id"].duplicated(keep=False)].sort_values("slide_id")
        raise ValueError(f"slide_id 重复，请先处理命名冲突:\n{dup[['slide_id', 'wsi_path']]}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["label", "patient_id", "slide_id"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("Saved:", out_csv)
    print("label counts:", df["label_text"].value_counts().to_dict())
    print("patient counts by label:")
    print(df.groupby("label_text")["patient_id"].nunique())


if __name__ == "__main__":
    main()