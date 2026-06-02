from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.options import Options
from core.transformer.decoder import AutoregressiveMeshDecoder, SegmentLatentBuilder, SegmentLatentDownsampler
from core.transformer.point import PointConditionEncoder


class LMM(nn.Module):
    """Hierarchical latent + autoregressive mesh token model.

    For a selected segment ``s``:
    - all ten segment latents are built from the point-cloud condition;
    - latents ``[0, ..., s-1]`` are placed before BOS as the autoregressive prefix;
    - latent ``s`` is injected after every causal self-attention layer through
      cross-attention, which makes the decoder specialize to the current segment;
    - future latents are never provided to the decoder.
    """

    def __init__(self, opt: Options, vocab_size: int):
        super().__init__()
        if opt.cond_mode != "point":
            raise ValueError("This cleaned training path currently supports cond_mode='point'.")
        if opt.hidden_dim % opt.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        point_stage_dims = (opt.point_encoder_dim, opt.point_encoder_dim_mid, opt.point_encoder_dim_out)
        point_stage_heads = (opt.point_encoder_heads, opt.point_encoder_heads_mid, opt.point_encoder_heads_out)
        for stage, (dim, heads) in enumerate(zip(point_stage_dims, point_stage_heads)):
            if dim % heads != 0:
                raise ValueError(f"point encoder stage {stage}: dim={dim} must be divisible by heads={heads}")
        if opt.point_encoder_dim_out % opt.num_heads != 0:
            raise ValueError("point_encoder_dim_out must be divisible by num_heads for segment latents")

        self.opt = opt
        self.vocab_size = vocab_size
        self.point_feature_dim = opt.point_encoder_dim_out
        self.point_encoder = PointConditionEncoder(
            point_encoder_dim=opt.point_encoder_dim,
            hidden_dim=opt.point_encoder_dim_out,
            num_heads=opt.point_encoder_heads,
            latent_size=opt.point_latent_size,
            num_layers=opt.point_encoder_layers,
            dropout=opt.dropout,
            checkpointing=opt.checkpointing,
            point_encoder_dim_mid=opt.point_encoder_dim_mid,
            point_encoder_dim_out=opt.point_encoder_dim_out,
            num_heads_mid=opt.point_encoder_heads_mid,
            num_heads_out=opt.point_encoder_heads_out,
            latent_size_mid=opt.point_latent_size_mid,
            latent_size_out=opt.point_latent_size_out,
        )
        self.segment_latents = SegmentLatentBuilder(
            num_segments=opt.num_segments,
            latent_tokens=opt.segment_latent_tokens,
            hidden_dim=self.point_feature_dim,
            num_heads=opt.num_heads,
            num_cross_layers=1,
            num_stage_layers=opt.segment_latent_layers,
            ffn_mult=opt.ffn_mult,
            dropout=opt.dropout,
            checkpointing=opt.checkpointing,
            max_seq_length=opt.max_seq_length,
        )
        self.get_segment_latents = SegmentLatentDownsampler(
            num_segments=opt.num_segments,
            latent_tokens=opt.segment_latent_tokens,
            hidden_dim=self.point_feature_dim,
            num_heads=opt.num_heads,
            num_cross_layers=1,
            ffn_mult=opt.ffn_mult,
            dropout=opt.dropout,
            checkpointing=opt.checkpointing,
            max_seq_length=opt.max_seq_length,
        )
        self.decoder = AutoregressiveMeshDecoder(
            vocab_size=vocab_size,
            hidden_dim=opt.hidden_dim,
            num_heads=opt.num_heads,
            num_layers=opt.num_layers,
            max_segment_length=opt.max_segment_length,
            pad_token_id=opt.pad_token_id,
            ffn_mult=opt.ffn_mult,
            dropout=opt.dropout,
            checkpointing=opt.checkpointing,
        )

        # self.up_scale_pre = nn.Linear(self.point_feature_dim, opt.hidden_dim, bias=False)
        # self.up_scale_cur = nn.Linear(self.point_feature_dim, opt.hidden_dim, bias=False)

    def encode_condition(self, points: torch.Tensor, total_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond_tokens = self.point_encoder(points)
        return self.segment_latents(cond_tokens, total_tokens), cond_tokens

    def _select_prefix_and_current(
        self,
        latents: torch.Tensor,
        segment_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_segments, latent_len, hidden = latents.shape
        max_prefix_len = (num_segments - 1) * latent_len
        flat = latents.reshape(bsz, num_segments * latent_len, hidden)
        prefix = flat[:, :max_prefix_len]

        prefix_lengths = segment_idx.clamp(0, num_segments - 1) * latent_len
        positions = torch.arange(max_prefix_len, device=latents.device).view(1, -1)
        prefix_mask = positions < prefix_lengths.view(-1, 1)
        prefix = prefix * prefix_mask.unsqueeze(-1).to(prefix.dtype)

        current = latents[torch.arange(bsz, device=latents.device), segment_idx]
        return prefix, prefix_mask, current

    def forward(self, data: dict) -> tuple[torch.Tensor, dict]:
        points = data["points"]
        input_ids = data["input_ids"]
        labels = data["labels"]
        token_attention_mask = data["token_attention_mask"]
        segment_idx = data["segment_idx"]
        total_tokens = data["total_tokens"]

        latents, cond_latents = self.encode_condition(points, total_tokens)
        prefix, prefix_mask, current = self._select_prefix_and_current(latents, segment_idx)

        current = self.get_segment_latents(segment_idx,cond_latents)
        
        # prefix = self.up_scale_pre(prefix)
        # cond_latents = self.up_scale_cur(cond_latents)

        logits, _ = self.decoder(
            input_ids=input_ids,
            current_latent=current,
            token_attention_mask=token_attention_mask,
            prefix_embeds=prefix,
            prefix_attention_mask=prefix_mask,
            use_cache=False,
        )

        prefix_len = prefix.shape[1]
        token_logits = logits[:, prefix_len : prefix_len + input_ids.shape[1], :]
        loss = F.cross_entropy(
            token_logits.reshape(-1, token_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

        with torch.no_grad():
            valid = labels != -100
            pred = token_logits.argmax(dim=-1)
            token_acc = ((pred == labels) & valid).sum().float() / valid.sum().clamp_min(1).float()
            first_token_acc = (pred[:, 0] == labels[:, 0]).float().mean()

        stats = {
            "loss": loss.detach(),
            "token_acc": token_acc.detach(),
            "first_token_acc": first_token_acc.detach(),
            "mean_segment_idx": segment_idx.float().mean().detach(),
            "mean_segment_len": token_attention_mask.sum(dim=1).float().mean().detach(),
        }
        return loss, stats

    @torch.no_grad()
    def generate_segment(
        self,
        points: torch.Tensor,
        total_tokens: torch.Tensor,
        segment_idx: int,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        generate_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate one segment with the exact conditioning path used in training."""
        generated = self.generate_segments(
            points=points,
            total_tokens=total_tokens,
            segment_indices=[int(segment_idx)],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            generate_mode=generate_mode,
        )
        return generated[int(segment_idx)]

    @torch.no_grad()
    def generate_segments(
        self,
        points: torch.Tensor,
        total_tokens: torch.Tensor,
        segment_indices: Sequence[int] | torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        generate_mode: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> dict[int, torch.Tensor]:
        """Generate multiple mesh segments in one autoregressive batch.

        The conditioning path is kept identical to training and to the original
        single-segment inference path:
            latents, cond_latents = encode_condition(points, total_tokens)
            prefix, prefix_mask, _ = _select_prefix_and_current(latents, segment_idx)
            current = get_segment_latents(segment_idx, cond_latents)

        ``segment_indices`` are decoded synchronously.  After a row predicts EOS,
        that row is removed from the active decoding batch, so later steps run
        with a smaller batch while preserving each row's KV cache.
        """
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if torch.is_tensor(segment_indices):
            segment_list = [int(x) for x in segment_indices.detach().cpu().tolist()]
        else:
            segment_list = [int(x) for x in segment_indices]
        if len(segment_list) == 0:
            raise ValueError("segment_indices must contain at least one segment")
        if len(set(segment_list)) != len(segment_list):
            raise ValueError(f"segment_indices must be unique, got {segment_list}")
        for idx in segment_list:
            if not 0 <= idx < self.opt.num_segments:
                raise ValueError(f"segment_idx must be in [0, {self.opt.num_segments - 1}], got {idx}")

        batch_size = len(segment_list)
        segment_idx_tensor = torch.tensor(segment_list, dtype=torch.long, device=device)

        if points.ndim == 2:
            points = points.unsqueeze(0)
        if points.ndim != 3:
            raise ValueError(f"points must have shape [N, 3] or [B, N, 3], got {tuple(points.shape)}")
        points = points.to(device=device, dtype=dtype)

        if total_tokens.ndim == 0:
            total_tokens = total_tokens.view(1)
        total_tokens = total_tokens.to(device=device)
        if total_tokens.ndim != 1:
            raise ValueError(f"total_tokens must be a scalar or a 1-D tensor, got {tuple(total_tokens.shape)}")

        if points.shape[0] == 1:
            if total_tokens.shape[0] not in {1, batch_size}:
                raise ValueError(
                    f"total_tokens must contain one value or {batch_size} values, got {total_tokens.shape[0]}"
                )
            latents, cond_latents = self.encode_condition(points, total_tokens[:1])
            if batch_size > 1:
                latents = latents.expand(batch_size, -1, -1, -1)
                cond_latents = cond_latents.expand(batch_size, -1, -1)
        elif points.shape[0] == batch_size:
            if total_tokens.shape[0] == 1:
                total_tokens = total_tokens.expand(batch_size)
            elif total_tokens.shape[0] != batch_size:
                raise ValueError(
                    f"total_tokens must contain one value or {batch_size} values, got {total_tokens.shape[0]}"
                )
            latents, cond_latents = self.encode_condition(points, total_tokens)
        else:
            raise ValueError(
                f"points batch size must be 1 or match the number of requested segments "
                f"({batch_size}), got {points.shape[0]}"
            )

        if max_new_tokens is None:
            max_new_tokens = self.opt.max_new_tokens
        train_length_limit = max(1, int(self.opt.max_segment_length))
        if max_new_tokens is None or int(max_new_tokens) <= 0:
            max_new_tokens = train_length_limit
        else:
            max_new_tokens = min(int(max_new_tokens), train_length_limit)
        temperature = float(self.opt.temperature if temperature is None else temperature)
        top_k = int(self.opt.top_k if top_k is None else top_k)
        generate_mode = generate_mode or self.opt.generate_mode

        prefix, prefix_mask, _ = self._select_prefix_and_current(latents, segment_idx_tensor)
        current = self.get_segment_latents(segment_idx_tensor, cond_latents)

        # State is tracked by original output slot.  Active rows store indices into
        # these lists, making row removal after EOS cheap and explicit.
        output_tokens: list[list[int]] = [[] for _ in segment_list]
        states: list[dict] = [{"counter": 0} for _ in segment_list]
        active_slots = torch.arange(batch_size, dtype=torch.long, device=device)

        past_key_values = None
        input_ids = torch.full((batch_size, 1), self.opt.bos_token_id, dtype=torch.long, device=device)
        token_positions = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        for step in range(max_new_tokens):
            active_bsz = int(active_slots.numel())
            if active_bsz == 0:
                break

            if past_key_values is None:
                prefix_embeds = prefix
                prefix_attention_mask = prefix_mask
                attention_mask = torch.ones((active_bsz, 1), dtype=torch.bool, device=device)
            else:
                prefix_embeds = None
                prefix_attention_mask = None
                # Full KV mask: [training prefix mask, BOS/token history mask].
                # ``history_len`` includes BOS and all generated tokens already in
                # the cache for the active rows.
                history_len = step + 1
                attention_mask = torch.cat(
                    [
                        prefix_mask,
                        torch.ones((active_bsz, history_len), dtype=torch.bool, device=device),
                    ],
                    dim=1,
                )

            logits, past_key_values = self.decoder(
                input_ids=input_ids,
                current_latent=current,
                token_attention_mask=attention_mask,
                prefix_embeds=prefix_embeds,
                prefix_attention_mask=prefix_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                token_positions=token_positions,
            )
            next_logits = logits[:, -1, :]
            generated_histories = [output_tokens[int(slot)] for slot in active_slots.detach().cpu().tolist()]
            active_states = [states[int(slot)] for slot in active_slots.detach().cpu().tolist()]
            next_tokens, active_states = self._sample_next_batch(
                generated_histories,
                active_states,
                next_logits,
                generate_mode,
                temperature,
                top_k,
            )

            active_slot_list = active_slots.detach().cpu().tolist()
            next_token_list = next_tokens.detach().cpu().tolist()
            for slot, token_id, new_state in zip(active_slot_list, next_token_list, active_states):
                slot = int(slot)
                output_tokens[slot].append(int(token_id))
                states[slot] = new_state

            if progress_callback is not None:
                finished_after_step = sum(
                    1
                    for tokens in output_tokens
                    if len(tokens) > 0 and tokens[-1] == self.opt.eos_token_id
                )
                progress_callback(active_bsz, active_bsz, finished_after_step)

            keep_mask = next_tokens != self.opt.eos_token_id
            if not bool(keep_mask.any()):
                active_slots = active_slots[:0]
                break

            if not bool(keep_mask.all()):
                keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
                active_slots = active_slots.index_select(0, keep_idx)
                prefix = prefix.index_select(0, keep_idx)
                prefix_mask = prefix_mask.index_select(0, keep_idx)
                current = current.index_select(0, keep_idx)
                input_ids = next_tokens.index_select(0, keep_idx).view(-1, 1)
                token_positions = torch.full(
                    (int(keep_idx.numel()), 1),
                    step + 1,
                    dtype=torch.long,
                    device=device,
                )
                past_key_values = self._index_past_key_values(past_key_values, keep_idx)
            else:
                input_ids = next_tokens.view(-1, 1)
                token_positions = torch.full(
                    (active_bsz, 1),
                    step + 1,
                    dtype=torch.long,
                    device=device,
                )

        if was_training:
            self.train()

        return {
            segment_idx: torch.tensor(tokens, dtype=torch.long, device=device)
            for segment_idx, tokens in zip(segment_list, output_tokens)
        }

    @staticmethod
    def _index_past_key_values(
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]],
        keep_idx: torch.Tensor,
    ) -> Optional[list[tuple[torch.Tensor, torch.Tensor]]]:
        if past_key_values is None:
            return None
        return [
            (key.index_select(0, keep_idx), value.index_select(0, keep_idx))
            for key, value in past_key_values
        ]

    def prefix_allowed_tokens_fn_with_state(self, input_ids, state: dict):
        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)

        idx = input_ids.shape[0]
        # print(f'=== prefix idx: {idx} ===')

        # BOS is always provided, so the first token must be BOM
        # 0=PAD, 1=BOS, 2=EOS, 3=L, 4=R, 5=BOM, 6~=coords
        if idx == 0: return [5], state

        # update state based on the last token
        if input_ids[-1] == 5:
            state['counter'] = 9 # after BOM, there must be 9 coords tokens
        elif input_ids[-1] in [3, 4]:
            state['counter'] = 3 # after LR, there must be 3 coords tokens
        elif input_ids[-1] >= 6:
            state['counter'] -= 1 # after coords, counter -1
        
        # set rules for the next token
        # counter > 0 means there are still coords to be filled
        if state['counter'] > 0:
            return list(range(6, self.vocab_size)), state
        # otherwise, it could be L/R/BOM/EOS
        else:
            return [3, 4, 5, self.opt.eos_token_id], state
        
    #@staticmethod
    def _sample_next(self, generated_ids, state: dict, logits: torch.Tensor, mode: str, temperature: float, top_k: int) -> torch.Tensor:
        if mode == "greedy" or temperature <= 0:
            return logits.argmax(dim=-1), state
        logits = logits / max(temperature, 1e-6)
        enable_value, state = self.prefix_allowed_tokens_fn_with_state(generated_ids, state)
        logits = mask_logits_by_enable_value(logits, enable_value)
        # Keep the original inference semantics: top_k is accepted by the CLI but
        # not applied here because the existing single-segment path left it off.
        # if top_k > 0 and top_k < logits.shape[-1]:
        #     values, _ = torch.topk(logits, top_k, dim=-1)
        #     threshold = values[:, -1].unsqueeze(-1)
        #     logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1), state

    def _sample_next_batch(
        self,
        generated_ids_batch: Sequence[Sequence[int]],
        states: list[dict],
        logits: torch.Tensor,
        mode: str,
        temperature: float,
        top_k: int,
    ) -> tuple[torch.Tensor, list[dict]]:
        if mode == "greedy" or temperature <= 0:
            return logits.argmax(dim=-1), states

        logits = logits / max(temperature, 1e-6)
        keep_mask = torch.zeros_like(logits, dtype=torch.bool)
        new_states: list[dict] = []
        for row, (generated_ids, state) in enumerate(zip(generated_ids_batch, states)):
            enable_value, state = self.prefix_allowed_tokens_fn_with_state(generated_ids, state)
            keep_mask[row, torch.as_tensor(enable_value, device=logits.device, dtype=torch.long)] = True
            new_states.append(state)
        logits = logits.masked_fill(~keep_mask, torch.finfo(logits.dtype).min)
        # Keep top_k disabled for parity with the existing single-segment sampler.
        # if top_k > 0 and top_k < logits.shape[-1]:
        #     values, _ = torch.topk(logits, top_k, dim=-1)
        #     threshold = values[:, -1].unsqueeze(-1)
        #     logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1), new_states

def mask_logits_by_enable_value(logits, enable_value):
    enable_idx = torch.as_tensor(enable_value, device=logits.device, dtype=torch.long)

    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
    keep_mask[:, enable_idx] = True

    return logits.masked_fill(
        ~keep_mask,
        torch.finfo(logits.dtype).min
    )