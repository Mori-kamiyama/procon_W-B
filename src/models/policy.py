from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


@dataclass
class ModelConfig:
    size: int
    vocab_size: int
    num_ops: int
    embed_dim: int = 32


class PolicyCNN(nn.Module):
    def __init__(self, cfg: ModelConfig, hidden_channels: int = 64, depth: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.size = cfg.size
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        layers = []
        in_ch = cfg.embed_dim
        for i in range(depth):
            out_ch = hidden_channels
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.tower = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, cfg.num_ops),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: (B, H, W) long
        x = self.embed(grid)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.tower(x)
        logits = self.head(x)
        return logits


class RetentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = d_model

        self.norm_attn = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, num_heads * head_dim)
        self.k_proj = nn.Linear(d_model, num_heads * head_dim)
        self.v_proj = nn.Linear(d_model, num_heads * head_dim)
        self.g_proj = nn.Linear(d_model, num_heads * head_dim)
        self.decay_logit = nn.Parameter(torch.zeros(num_heads))
        self.out_proj = nn.Linear(num_heads * head_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        y = self.norm_attn(x)
        B, L, _ = y.shape
        q = self.q_proj(y).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(y).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(y).view(B, L, self.num_heads, self.head_dim)
        g = torch.sigmoid(self.g_proj(y)).view(B, L, self.num_heads, self.head_dim)

        decay = torch.sigmoid(self.decay_logit).view(1, self.num_heads, 1)
        state = torch.zeros(B, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        outputs = []
        scale = self.head_dim ** -0.5
        for t in range(L):
            state = decay * state + k[:, t] * v[:, t]
            outputs.append(g[:, t] * q[:, t] * state * scale)
        o = torch.stack(outputs, dim=1).reshape(B, L, self.num_heads * self.head_dim)
        o = self.out_proj(o)
        x = residual + self.dropout(o)

        ff_residual = x
        ff_out = self.ff(self.norm_ff(x))
        return ff_residual + self.dropout(ff_out)


class PolicyRetNet(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        head_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if head_dim is None:
            head_dim = d_model // num_heads
        if head_dim * num_heads != d_model:
            raise ValueError("d_model must be divisible by num_heads")
        self.size = cfg.size
        seq_len = cfg.size * cfg.size
        self.embed = nn.Embedding(cfg.vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        init.trunc_normal_(self.pos_embed, std=0.02)
        init.trunc_normal_(self.cls_token, std=0.02)
        blocks = [
            RetentionBlock(d_model=d_model, num_heads=num_heads, head_dim=head_dim, dropout=dropout)
            for _ in range(num_layers)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, cfg.num_ops)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: (B, H, W)
        B, H, W = grid.shape
        tokens = self.embed(grid.view(B, -1))
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits
