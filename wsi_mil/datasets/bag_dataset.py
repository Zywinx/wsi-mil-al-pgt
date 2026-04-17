import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from wsi_mil.utils.io import read_json

@dataclass
class BagItem:
    slide_id: str
    label: int
    patient_id: str
    tile_records: List[Dict[str, Any]]

def build_transforms(train: bool, img_size: int = 224):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

class SlideBagDataset(Dataset):
    """
    full_bag: 是否返回该 WSI 的所有图块
    lazy_load: 如果 True，__getitem__ 只返回 tile_records 路径列表，不加载图像
               用于推理时外部自行分块加载，避免 OOM
    """
    def __init__(
        self,
        splits_csv: str,
        bag_index_json: str,
        split: str,
        bag_size: int = 256,
        img_size: int = 224,
        seed: int = 7,
        full_bag: bool = False,
        lazy_load: bool = False,  # [新增]: 延迟加载模式，返回路径而非图像
    ):
        self.df = pd.read_csv(splits_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.bags: Dict[str, List[Dict[str, Any]]] = read_json(bag_index_json)
        self.split = split
        self.bag_size = bag_size
        self.full_bag = full_bag
        self.lazy_load = lazy_load  # [新增]: 保存状态
        self.rng = random.Random(seed)
        self.transform = build_transforms(train=(split == "train"), img_size=img_size)
        self.img_size = img_size  # [新增]: 保存用于 lazy_load 模式

        self.items: List[BagItem] = []
        for _, r in self.df.iterrows():
            slide_id = str(r["slide_id"])
            if slide_id not in self.bags:
                continue
            self.items.append(
                BagItem(
                    slide_id=slide_id,
                    label=int(r["label"]),
                    patient_id=str(r.get("patient_id", "")),
                    tile_records=self.bags[slide_id],
                )
            )

    def __len__(self):
        return len(self.items)

    def _sample_tile_records(self, tile_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(tile_records) == 0:
            logging.warning(f"Empty bag detected with 0 tiles for slide")
            return []
        if len(tile_records) >= self.bag_size:
            return self.rng.sample(tile_records, self.bag_size)
        out = []
        for _ in range(self.bag_size):
            out.append(self.rng.choice(tile_records))
        return out
        
    def _fixed_tile_records(self, tile_records):
        fixed = sorted(
            tile_records,
            key=lambda r: (int(r["y"]), int(r["x"]), str(r["tile_path"]))
        )
        if len(fixed) >= self.bag_size:
            return fixed[:self.bag_size]
        reps = (self.bag_size + len(fixed) - 1) // len(fixed)
        return (fixed * reps)[:self.bag_size]

    def __getitem__(self, idx: int):
        item = self.items[idx]

        # === [修改逻辑]: 根据 full_bag 和 split 决定采样策略 ===
        if self.full_bag:
            # 评估/推理模式下，使用确定性排序的所有切块
            sampled = sorted(
                item.tile_records,
                key=lambda r: (int(r["y"]), int(r["x"]), str(r["tile_path"]))
            )
        elif self.split == "train":
            # 训练模式下，进行随机采样或补齐
            sampled = self._sample_tile_records(item.tile_records)
        else:
            # 如果 full_bag=False 且处于非 train 模式，退化为固定的截断采样
            sampled = self._fixed_tile_records(item.tile_records)
        # ==============================================================

        # [新增]: lazy_load 模式，只返回路径列表，不加载图像
        if self.lazy_load:
            tile_paths = [rec["tile_path"] for rec in sampled]
            coords = [(int(rec["x"]), int(rec["y"])) for rec in sampled]
            label = torch.tensor(item.label, dtype=torch.long)
            meta = {
                "slide_id": item.slide_id,
                "patient_id": item.patient_id,
                "tile_paths": tile_paths,
                "coords": coords,
                "tile_records": sampled,  # 保留完整记录供外部使用
                "n_tiles_total": len(item.tile_records),  # 新增：总tile数量
            }
            # 返回空 tensor 作为占位，实际图像在外部分块加载
            return torch.empty(0), label, meta

        # 正常模式：加载所有图像
        imgs = []
        tile_paths = []
        coords = []

        for rec in sampled:
            tile_path = rec["tile_path"]
            img = Image.open(tile_path).convert("RGB")
            img = self.transform(img)
            imgs.append(img)
            tile_paths.append(tile_path)
            coords.append((int(rec["x"]), int(rec["y"])))

        bag_imgs = torch.stack(imgs, dim=0)
        label = torch.tensor(item.label, dtype=torch.long)
        meta = {
            "slide_id": item.slide_id,
            "patient_id": item.patient_id,
            "tile_paths": tile_paths,
            "coords": coords,
            "n_tiles_total": len(item.tile_records),  # 新增：总tile数量
        }
        return bag_imgs, label, meta