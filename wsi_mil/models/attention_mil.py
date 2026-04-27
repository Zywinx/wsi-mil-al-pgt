import torch
import torch.nn as nn

class AttentionMIL(nn.Module):
    """
    Implements:
      e_i = w^T tanh(V z_i)
      alpha = softmax(e)
      h = sum_i alpha_i z_i
      logit = Linear(h)
    """
    def __init__(self, in_dim: int, attn_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, z: torch.Tensor):
        # z: [B,N,D]
        a = torch.tanh(self.V(z))          # [B,N,attn_dim]
        e = self.w(a).squeeze(-1)          # [B,N]
        alpha = torch.softmax(e, dim=1)    # [B,N]
        h = torch.sum(alpha.unsqueeze(-1) * z, dim=1)   # [B,D]
        logit = self.classifier(h).squeeze(-1)          # [B]
        return logit, alpha, h, e