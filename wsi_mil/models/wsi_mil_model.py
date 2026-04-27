from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from wsi_mil.models.tile_encoder import TileEncoder
from wsi_mil.models.attention_mil import AttentionMIL


@dataclass
class ForwardOut:
    slide_logit: torch.Tensor  # [B]
    slide_prob: torch.Tensor   # [B]
    alpha: torch.Tensor        # [B,N]
    h: torch.Tensor            # [B,D]
    tile_z: torch.Tensor       # [B,N,D]
    tile_logit: Optional[torch.Tensor] = None
    tile_prob: Optional[torch.Tensor] = None
    tile_u: Optional[torch.Tensor] = None


class WSIBaselineMIL(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        attn_dim: int = 256,
        encoder_pretrained: bool = True,
        encoder_dropout: float = 0.0,
        mil_dropout: float = 0.25,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = TileEncoder(
            out_dim=embed_dim,
            pretrained=encoder_pretrained,
            dropout=encoder_dropout,
        )
        self.mil = AttentionMIL(
            in_dim=embed_dim,
            attn_dim=attn_dim,
            dropout=mil_dropout,
        )

    def forward(self, bag_imgs: torch.Tensor) -> ForwardOut:
        # bag_imgs: [B,N,3,H,W]
        B, N = bag_imgs.shape[:2]
        x = bag_imgs.reshape(B * N, *bag_imgs.shape[2:])
        z = self.encoder(x)         # [B*N,D]
        z = z.view(B, N, -1)        # [B,N,D]
        slide_logit, alpha, h, _ = self.mil(z)
        slide_prob = torch.sigmoid(slide_logit)
        return ForwardOut(
            slide_logit=slide_logit,
            slide_prob=slide_prob,
            alpha=alpha,
            h=h,
            tile_z=z,
        )
