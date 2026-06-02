from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from core.transformer.attention import CausalSelfAttention, CrossAttention, FeedForward


class CausalTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4, dropout: float = 0.0, checkpointing: bool = True):
        super().__init__()
        self.checkpointing = checkpointing
        self.ln_self = nn.LayerNorm(dim)
        self.self_attn = CausalSelfAttention(dim, num_heads, dropout=dropout)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mult=ffn_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        if self.training and self.checkpointing and past_key_value is None and not use_cache:
            return checkpoint(self._forward_no_cache, x, attention_mask, use_reentrant=False), None
        return self._forward(x, attention_mask, past_key_value, use_cache)

    def _forward_no_cache(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        out, _ = self._forward(x, attention_mask, None, False)
        return out

    def _forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present = self.self_attn(
            self.ln_self(x),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            use_flash=self.training,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_mlp(x))
        return x, present


class CrossTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4, dropout: float = 0.0, checkpointing: bool = True):
        super().__init__()
        self.checkpointing = checkpointing
        self.ln_q = nn.LayerNorm(dim)
        self.ln_ctx = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout=dropout)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask_q: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.training and self.checkpointing:
            return checkpoint(self._forward, x, context, mask_q, use_reentrant=False)
        return self._forward(x, context, mask_q)

    def _forward(self, x: torch.Tensor, context: torch.Tensor, mask_q: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.cross_attn(
            self.ln_q(x),
            self.ln_ctx(context),
            mask_q=mask_q,
            mask_kv=None,
            use_flash=self.training,
        )
        x = x + self.mlp(self.ln_mlp(x))
        return x


class SegmentLatentBuilder(nn.Module):
    """Build 10 hierarchical segment latents from global 3D condition tokens.

    The causal stage regressor is the cleaned-up form of the old
    ``stage_regress`` idea: after each stage latent attends to the global 3D
    condition, the flattened stage sequence is processed causally, so stage i
    can depend on stages <= i and never on future stages.
    """

    def __init__(
        self,
        num_segments: int,
        latent_tokens: int,
        hidden_dim: int,
        num_heads: int,
        num_cross_layers: int = 1,
        num_stage_layers: int = 1,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        checkpointing: bool = True,
        max_seq_length: int = 40960,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.latent_tokens = latent_tokens
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        self.base_query = nn.Parameter(torch.randn(1, 1, latent_tokens, hidden_dim) / math.sqrt(hidden_dim))
        self.stage_embed = nn.Embedding(num_segments, hidden_dim)
        self.latent_pos_embed = nn.Embedding(latent_tokens, hidden_dim)
        self.length_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cross_layers = nn.ModuleList(
            [CrossTransformerBlock(hidden_dim, num_heads, ffn_mult, dropout, checkpointing) for _ in range(num_cross_layers)]
        )
        self.stage_regress = nn.ModuleList(
            [CausalTransformerBlock(hidden_dim, num_heads, ffn_mult, dropout, checkpointing) for _ in range(num_stage_layers)]
        )
        # self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, cond_tokens: torch.Tensor, total_tokens: torch.Tensor) -> torch.Tensor:
        bsz = cond_tokens.shape[0]
        device = cond_tokens.device
        dtype = cond_tokens.dtype

        stage_ids = torch.arange(self.num_segments, device=device)
        latent_pos = torch.arange(self.latent_tokens, device=device)
        stage_emb = self.stage_embed(stage_ids).view(1, self.num_segments, 1, self.hidden_dim)
        pos_emb = self.latent_pos_embed(latent_pos).view(1, 1, self.latent_tokens, self.hidden_dim)

        total = total_tokens.to(device=device, dtype=torch.float32).clamp_min(1.0)
        length_features = torch.stack(
            [
                torch.log1p(total) / math.log1p(float(max(1, self.max_seq_length))),
                (total / float(max(1, self.max_seq_length))).clamp(max=4.0),
            ],
            dim=-1,
        ).to(dtype)
        length_emb = self.length_mlp(length_features).to(dtype).view(bsz, 1, 1, self.hidden_dim)

        x = self.base_query.to(dtype=dtype, device=device) + stage_emb + pos_emb + length_emb
        x = x.reshape(bsz, self.num_segments * self.latent_tokens, self.hidden_dim)

        for layer in self.cross_layers:
            x = layer(x, cond_tokens)
        for layer in self.stage_regress:
            x, _ = layer(x, attention_mask=None, past_key_value=None, use_cache=False)

        # x = self.out_norm(x)
        return x.view(bsz, self.num_segments, self.latent_tokens, self.hidden_dim)


# Backward-compatible name for the old CrossEmbed concept.
CrossEmbed = SegmentLatentBuilder

class SegmentLatentDownsampler(nn.Module):

    def __init__(
        self,
        num_segments: int,
        latent_tokens: int,
        hidden_dim: int,
        num_heads: int,
        num_cross_layers: int = 1,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        checkpointing: bool = True,
        max_seq_length: int = 40960,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.latent_tokens = 512
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        self.stage_embed = nn.Embedding(self.num_segments+3, hidden_dim)
        self.latent_pos_embed = nn.Embedding(self.latent_tokens+3, hidden_dim)

        self.cross_layers = nn.ModuleList(
            [CrossTransformerBlock(hidden_dim, num_heads, ffn_mult, dropout, checkpointing) for _ in range(num_cross_layers)]
        )

    def forward(self, stage_ids, cond_tokens: torch.Tensor) -> torch.Tensor:
        bsz = cond_tokens.shape[0]
        device = cond_tokens.device
        dtype = cond_tokens.dtype

        stage_emb = self.stage_embed(stage_ids).view(bsz, 1, self.hidden_dim)
        latent_pos = torch.arange(self.latent_tokens, device=device)
        pos_emb = self.latent_pos_embed(latent_pos).unsqueeze(0)
        stage_emb = stage_emb+pos_emb
        # x = x.reshape(bsz, self.num_segments * self.latent_tokens, self.hidden_dim)

        for layer in self.cross_layers:
            x = layer(stage_emb, cond_tokens)

        # x = self.out_norm(x)
        return x

class MeshDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4, dropout: float = 0.0, checkpointing: bool = True):
        super().__init__()
        self.checkpointing = checkpointing
        self.ln_self = nn.LayerNorm(dim)
        self.self_attn = CausalSelfAttention(dim, num_heads, dropout=dropout)
        self.ln_cross_q = nn.LayerNorm(dim)
        self.ln_cross_ctx = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout=dropout)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mult=ffn_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        current_latent: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        if self.training and self.checkpointing and past_key_value is None and not use_cache:
            return checkpoint(self._forward_no_cache, x, current_latent, attention_mask, use_reentrant=False), None
        return self._forward(x, current_latent, attention_mask, past_key_value, use_cache)

    def _forward_no_cache(
        self,
        x: torch.Tensor,
        current_latent: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out, _ = self._forward(x, current_latent, attention_mask, None, False)
        return out

    def _forward(
        self,
        x: torch.Tensor,
        current_latent: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        self_out, present = self.self_attn(
            self.ln_self(x),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            use_flash=self.training,
        )
        x = x + self_out
        x = x + self.cross_attn(
            self.ln_cross_q(x),
            self.ln_cross_ctx(current_latent), 
            mask_q=attention_mask if past_key_value is None else None,
            mask_kv=None,
            use_flash=self.training and past_key_value is None,
        )
        x = x + self.mlp(self.ln_mlp(x))
        return x, present


class AutoregressiveMeshDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_segment_length: int,
        pad_token_id: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        checkpointing: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.token_pos_embed = nn.Embedding(max_segment_length + 2, hidden_dim)
        self.layers = nn.ModuleList(
            [MeshDecoderBlock(hidden_dim, num_heads, ffn_mult, dropout, checkpointing) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        current_latent: torch.Tensor,
        token_attention_mask: Optional[torch.Tensor] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        bsz, token_len = input_ids.shape
        device = input_ids.device

        if token_positions is None:
            token_positions = torch.arange(token_len, device=device).view(1, token_len).repeat(bsz, 1)
        token_positions = token_positions.clamp(max=self.token_pos_embed.num_embeddings - 1)
        token_embeds = self.token_embed(input_ids) + self.token_pos_embed(token_positions)

        if prefix_embeds is not None:
            prefix_len = prefix_embeds.shape[1]
            if prefix_attention_mask is None:
                prefix_attention_mask = torch.ones(bsz, prefix_len, dtype=torch.bool, device=device)
            if token_attention_mask is None:
                token_attention_mask = torch.ones(bsz, token_len, dtype=torch.bool, device=device)
            hidden_states = torch.cat([prefix_embeds, token_embeds], dim=1)
            attention_mask = torch.cat([prefix_attention_mask, token_attention_mask], dim=1)
        else:
            hidden_states = token_embeds
            attention_mask = token_attention_mask

        # During cached decoding there is no padding and the old valid tokens are
        # already inside the cache, so a current-step mask is unnecessary.
        if past_key_values is not None and attention_mask is not None and attention_mask.shape[1] == token_len:
            attention_mask_for_layers = None
        else:
            attention_mask_for_layers = attention_mask

        next_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states,
                current_latent=current_latent,
                attention_mask=attention_mask_for_layers,
                past_key_value=past,
                use_cache=use_cache,
            )
            if use_cache:
                assert present is not None
                next_cache.append(present)

        logits = self.lm_head(self.final_norm(hidden_states))
        return logits, (next_cache if use_cache else None)
