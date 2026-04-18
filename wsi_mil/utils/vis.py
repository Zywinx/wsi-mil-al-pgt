from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_topk_mosaic(tile_paths: List[str], scores: List[float], out_png: str, thumb_size: int = 224):
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    n = len(tile_paths)
    cols = min(8, n)
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(cols * 2, rows * 2))
    for i, (p, s) in enumerate(zip(tile_paths, scores)):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = Image.open(p).convert("RGB").resize((thumb_size, thumb_size))
        ax.imshow(img)
        ax.set_title(f"{s:.3f}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)