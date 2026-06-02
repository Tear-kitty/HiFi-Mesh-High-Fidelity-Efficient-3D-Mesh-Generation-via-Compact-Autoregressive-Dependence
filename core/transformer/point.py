from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from core.transformer.attention import CrossAttention, FeedForward


class PointFourierEmbed(nn.Module):
    def __init__(self, freq_embed_dim: int = 48):
        super().__init__()
        if freq_embed_dim % 6 != 0:
            raise ValueError("freq_embed_dim must be divisible by 6")
        bands = torch.pow(2, torch.arange(freq_embed_dim // 6)).float() * math.pi
        basis = torch.stack(
            [
                torch.cat([bands, torch.zeros_like(bands), torch.zeros_like(bands)]),
                torch.cat([torch.zeros_like(bands), bands, torch.zeros_like(bands)]),
                torch.cat([torch.zeros_like(bands), torch.zeros_like(bands), bands]),
            ]
        )
        self.register_buffer("basis", basis)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        projected = torch.einsum("bnd,de->bne", points, self.basis.to(points.dtype))
        return torch.cat([projected.sin(), projected.cos(), points], dim=-1)


class CrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4, dropout: float = 0.0, checkpointing: bool = True):
        super().__init__()
        self.checkpointing = checkpointing
        self.ln_q = nn.LayerNorm(dim)
        self.ln_ctx = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads, dropout=dropout)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.training and self.checkpointing:
            return checkpoint(self._forward, x, context, use_reentrant=False)
        return self._forward(x, context)

    def _forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        use_flash = self.training
        x = x + self.attn(self.ln_q(x), self.ln_ctx(context), use_flash=use_flash)
        x = x + self.mlp(self.ln_mlp(x))
        return x


class PointConditionEncoder(nn.Module):
    """Encode a point cloud with staged token-count and feature-width growth.

    The intended default path is:
    - stage 0: 2048 tokens,  256 channels, attends to sampled point features;
    - stage 1: 3072 tokens,  768 channels, attends to stage-0 tokens;
    - stage 2: 4096 tokens, 1536 channels, attends to stage-1 tokens.

    The original public constructor arguments are kept as the first-stage knobs;
    the ``*_mid`` and ``*_out`` arguments restore the progressive extraction path.
    """

    def __init__(
        self,
        point_encoder_dim: int,
        hidden_dim: int,
        num_heads: int,
        latent_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        checkpointing: bool = True,
        point_encoder_dim_mid: Optional[int] = None,
        point_encoder_dim_out: Optional[int] = None,
        num_heads_mid: Optional[int] = None,
        num_heads_out: Optional[int] = None,
        latent_size_mid: Optional[int] = None,
        latent_size_out: Optional[int] = None,
    ):
        super().__init__()
        self.latent_sizes = (
            int(latent_size),
            int(latent_size_mid if latent_size_mid is not None else latent_size),
            int(latent_size_out if latent_size_out is not None else latent_size),
        )
        self.stage_dims = (
            int(point_encoder_dim),
            int(point_encoder_dim_mid if point_encoder_dim_mid is not None else hidden_dim),
            int(point_encoder_dim_out if point_encoder_dim_out is not None else hidden_dim),
        )
        self.stage_heads = (
            int(num_heads),
            int(num_heads_mid if num_heads_mid is not None else num_heads),
            int(num_heads_out if num_heads_out is not None else num_heads),
        )
        self.output_dim = self.stage_dims[-1]
        self.latent_size = self.latent_sizes[-1]

        if any(size <= 0 for size in self.latent_sizes):
            raise ValueError(f"all point latent sizes must be positive, got {self.latent_sizes}")
        if any(dim <= 0 for dim in self.stage_dims):
            raise ValueError(f"all point encoder dims must be positive, got {self.stage_dims}")
        if any(heads <= 0 for heads in self.stage_heads):
            raise ValueError(f"all point encoder heads must be positive, got {self.stage_heads}")
        for stage, (dim, heads) in enumerate(zip(self.stage_dims, self.stage_heads)):
            if dim % heads != 0:
                raise ValueError(f"point encoder stage {stage}: dim={dim} must be divisible by heads={heads}")

        self.point_embed = PointFourierEmbed(freq_embed_dim=48)
        self.point_proj = nn.Linear(51, self.stage_dims[0])

        self.queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, size, dim) / math.sqrt(dim))
                for size, dim in zip(self.latent_sizes, self.stage_dims)
            ]
        )
        self.stage_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossBlock(
                            dim,
                            heads,
                            ffn_mult=4,
                            dropout=dropout,
                            checkpointing=checkpointing,
                        )
                        for _ in range(num_layers)
                    ]
                )
                for dim, heads in zip(self.stage_dims, self.stage_heads)
            ]
        )
        self.stage_projs = nn.ModuleList(
            [
                nn.Sequential(
                    # nn.LayerNorm(self.stage_dims[idx - 1]),
                    nn.Linear(self.stage_dims[idx - 1], self.stage_dims[idx]),
                    # nn.GELU(),
                    # nn.LayerNorm(self.stage_dims[idx]),
                )
                for idx in range(1, len(self.stage_dims))
            ]
        )

    def _run_stage(self, stage_idx: int, context: torch.Tensor) -> torch.Tensor:
        query = self.queries[stage_idx]
        x = query.repeat(context.shape[0], 1, 1).to(dtype=context.dtype, device=context.device)
        for block in self.stage_blocks[stage_idx]:
            x = block(x, context)
        return x

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        context = self.point_proj(self.point_embed(points))
        x = None
        for stage_idx in range(len(self.stage_dims)):
            if stage_idx > 0:
                context = self.stage_projs[stage_idx - 1](x)

            x = self._run_stage(stage_idx, context)

        return x
