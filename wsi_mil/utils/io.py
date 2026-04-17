import json
from pathlib import Path

def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj, path: str):
    p = Path(path)
    mkdir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)