import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TileEncoder(nn.Module):
    def __init__(self, out_dim: int = 512, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.proj = nn.Identity() if out_dim == in_dim else nn.Linear(in_dim, out_dim)
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        feat = self.backbone(x)  # [B,in_dim]
        feat = self.do(feat)
        z = self.proj(feat)      # [B,out_dim]
        return z
