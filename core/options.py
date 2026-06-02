from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Options:
    """Project configuration.

    The defaults are intentionally conservative. The important research knobs are
    near the top so they can be overridden from ``main.py`` / command line.
    """

    # ------------------------- data / tokenizer -------------------------
    ply_paths_json: str = "dataset/ply_paths.json"
    use_meto: bool = True
    meto_backend: str = "LR_ABSCO"
    discrete_bins: int = 512
    vocab_size: Optional[int] = None

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    segment_boundary_token: int = 5

    num_segments: int = 10
    force_segment_start_token: bool = True
    strict_segment_boundary: bool = False
    max_seq_length: int = 200000
    max_segment_length: int = 25000

    train_split: float = 0.98
    repeat_per_epoch: int = 10
    point_num: int = 131072
    use_scale_aug: bool = False
    use_decimate_aug: bool = False
    decimate_prob: float = 0.25
    min_decimate_faces: int = 200

    # --------------------------- model shape ---------------------------
    cond_mode: str = "point"
    hidden_dim: int = 1536
    num_heads: int = 16
    num_layers: int = 24
    ffn_mult: int = 4
    dropout: float = 0.0

    point_encoder_dim: int = 256
    point_encoder_dim_mid: int = 768
    point_encoder_dim_out: int = 1536
    point_encoder_heads: int = 4
    point_encoder_heads_mid: int = 12
    point_encoder_heads_out: int = 24
    point_latent_size: int = 2048
    point_latent_size_mid: int = 3072
    point_latent_size_out: int = 4096
    point_encoder_layers: int = 1

    # This is the key knob requested in the task: latent tokens per segment.
    segment_latent_tokens: int = 64
    segment_latent_layers: int = 1

    # ---------------------------- training -----------------------------
    workspace: str = "./workspace"
    log_dir: str = "./log"
    batch_size: int = 6
    num_workers: int = 0
    gradient_accumulation_steps: int = 1
    num_epochs: int = 800
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    mixed_precision: str = "bf16"
    checkpointing: bool = True
    save_checkpoint: bool = True
    checkpointing_epoch: int = 10
    seed: int = 42
    resume: Optional[str] = None
    start_epoch: int = 0
    eval_every: int = checkpointing_epoch
    eval_batches: int = 1
    eval_batch:int = 1
    use_wandb: bool = False

    # ---------------------------- inference ----------------------------
    generate_mode: str = "sample"  # greedy | sample
    temperature: float = 1.0
    top_k: int = 0
    # ``None`` follows ``max_segment_length`` during inference so generation is
    # not capped below the segment length used in training.
    max_new_tokens: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Options":
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})
