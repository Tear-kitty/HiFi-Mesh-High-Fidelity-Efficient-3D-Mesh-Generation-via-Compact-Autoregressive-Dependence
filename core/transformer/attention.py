from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    FLASH_ATTN_AVAILABLE = True
except Exception:  # noqa: BLE001
    FLASH_ATTN_AVAILABLE = False


def _naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_q: Optional[torch.Tensor] = None,
    mask_kv: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    """Reference attention used for eval / generation.

    q, k, v are [B, T, H, D]. The causal mask supports KV-cache decoding
    where K/V length can be larger than Q length.
    """
    bsz, q_len, num_heads, head_dim = q.shape
    kv_len = k.shape[1]

    q = q.transpose(1, 2).reshape(bsz * num_heads, q_len, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, kv_len, head_dim)
    v = v.transpose(1, 2).reshape(bsz * num_heads, kv_len, head_dim)

    scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    scores = scores.view(bsz, num_heads, q_len, kv_len)

    if mask_kv is not None:
        kv_mask = mask_kv[:, None, None, :].to(torch.bool)
        scores = scores.masked_fill(~kv_mask, torch.finfo(scores.dtype).min)

    if causal:
        q_positions = torch.arange(kv_len - q_len, kv_len, device=scores.device)
        k_positions = torch.arange(kv_len, device=scores.device)
        causal_mask = k_positions[None, :] <= q_positions[:, None]
        scores = scores.masked_fill(~causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)

    attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
    if dropout > 0.0:
        attn = F.dropout(attn, p=dropout, training=True)

    attn = attn.view(bsz * num_heads, q_len, kv_len)
    out = torch.bmm(attn, v)
    out = out.view(bsz, num_heads, q_len, head_dim).transpose(1, 2).contiguous()

    if mask_q is not None:
        out = out * mask_q[:, :, None, None].to(out.dtype)
    return out


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_q: Optional[torch.Tensor] = None,
    mask_kv: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    causal: bool = False,
    use_flash: bool = True,
) -> torch.Tensor:
    """Attention wrapper.

    Training modules call this with ``use_flash=True``. During ``eval()`` the
    model passes ``use_flash=False`` so autoregressive inference uses ordinary
    attention and can rely on a Python-visible KV cache.
    """
    if not use_flash or not FLASH_ATTN_AVAILABLE:
        return _naive_attention(q, k, v, mask_q, mask_kv, dropout=dropout, causal=causal)

    bsz, q_len, num_heads, head_dim = q.shape
    kv_len = k.shape[1]

    if mask_q is None and mask_kv is None:
        return flash_attn_func(q, k, v, dropout, causal=causal)

    if mask_q is None:
        mask_q = torch.ones(bsz, q_len, dtype=torch.bool, device=q.device)
    if mask_kv is None:
        mask_kv = torch.ones(bsz, kv_len, dtype=torch.bool, device=q.device)

    q_unpad, indices_q, cu_q, max_q, _ = unpad_input(q, mask_q)
    k_unpad, indices_kv, cu_kv, max_kv, _ = unpad_input(k, mask_kv)
    v_unpad = index_first_axis(v.reshape(-1, num_heads, head_dim), indices_kv)

    out_unpad = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_kv,
        max_seqlen_q=max_q,
        max_seqlen_k=max_kv,
        dropout_p=dropout,
        causal=causal,
    )
    return pad_input(out_unpad, indices_q, bsz, q_len)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, gates = x.chunk(2, dim=-1)
        return values * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = min(dim * mult, 8192)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.context_dim = context_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(self.context_dim, dim)
        self.v_proj = nn.Linear(self.context_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask_q: Optional[torch.Tensor] = None,
        mask_kv: Optional[torch.Tensor] = None,
        use_flash: bool = True,
    ) -> torch.Tensor:
        bsz, q_len, _ = x.shape
        kv_len = context.shape[1]
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(bsz, kv_len, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(bsz, kv_len, self.num_heads, self.head_dim)
        out = flash_attention(q, k, v, mask_q, mask_kv, dropout=self.dropout if self.training else 0.0, causal=False, use_flash=use_flash)
        return self.out_proj(out.reshape(bsz, q_len, self.dim))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        use_flash: bool = True,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            # Cached decoding is eval-only and uses ordinary attention.
            mask_q = None
            mask_kv = attention_mask
            effective_flash = False
        else:
            mask_q = attention_mask
            mask_kv = attention_mask
            effective_flash = use_flash

        out = flash_attention(
            q,
            k,
            v,
            mask_q=mask_q,
            mask_kv=mask_kv,
            dropout=self.dropout if self.training else 0.0,
            causal=True,
            use_flash=effective_flash,
        )
        out = self.out_proj(out.reshape(bsz, seq_len, self.dim))
        present = (k, v) if use_cache else None
        return out, present
