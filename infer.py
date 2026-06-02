from __future__ import annotations
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse
import ast
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from core.models import LMM
from core.options import Options
from core.provider import load_tokenized_ply
from core.utils import (
    get_tokenizer,
    load_ply_paths,
    save_mesh_from_tokens,
    split_coords_into_segments,
    write_json,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Autoregressive inference for one or more HiFi-Mesh segments")
    parser.add_argument("--checkpoint", required=False, default='./workspace/checkpoint-0199', help="checkpoint .pth or checkpoint directory")
    parser.add_argument("--ply-path", default='./dataset/Mesh/16/16584be8afde41c4914046cd98426109.ply', help="absolute path to the conditioning PLY")
    parser.add_argument("--ply-index", type=int, default=None, help="fallback: index in opt.ply_paths_json")
    parser.add_argument("--segment-idx", type=int, required=False, default=0, help="backward-compatible fallback when --segment-indices is omitted")
    parser.add_argument(
        "--segment-indices",
        nargs="+",
        default="[0,1,2,3,4,5,6,7,8,9]",
        help='segments to generate together, e.g. "[0,1,3,4]", "0,1,3,4", or "0 1 3 4"',
    )
    parser.add_argument("--output-tokens", default=None)
    parser.add_argument("--output-mesh", default='./results/generated.ply', help="combined output mesh path")
    parser.add_argument("--output-segment-mesh-dir", default=None, help="directory for per-segment meshes; default: <output-mesh-stem>_segments")
    parser.add_argument("--reconstruct-with-gt-context", type=str2bool, default=False)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--generate-mode", choices=["greedy", "sample"], default='sample')
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", type=str2bool, default=True, help="run the model in bfloat16 on CUDA")
    parser.add_argument("--seed", type=int, default=None, help="override checkpoint seed for deterministic inference")
    parser.add_argument("--use-meto", type=str2bool, default=None, help="override checkpoint tokenizer setting")
    parser.add_argument("--ply-paths-json", default=None, help="override checkpoint ply_paths_json when using --ply-index")
    parser.add_argument("--progress", type=str2bool, default=True, help="show token/s progress during autoregressive inference")
    return parser.parse_args()


def parse_segment_indices(raw_indices, fallback_segment_idx: int) -> list[int]:
    if raw_indices is None:
        indices = [int(fallback_segment_idx)]
    else:
        if isinstance(raw_indices, (list, tuple)):
            text = " ".join(str(item) for item in raw_indices).strip()
        else:
            text = str(raw_indices).strip()
        if not text:
            raise ValueError("--segment-indices must contain at least one segment index")

        if text[0] in "[(":
            parsed = ast.literal_eval(text)
            if isinstance(parsed, int):
                indices = [parsed]
            else:
                indices = list(parsed)
        else:
            indices = [int(item) for item in text.replace(",", " ").split()]

    if len(indices) == 0:
        raise ValueError("At least one segment index is required")
    try:
        indices = [int(idx) for idx in indices]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"segment indices must be integers, got {indices!r}") from exc
    if len(set(indices)) != len(indices):
        raise ValueError(f"segment indices must be unique, got {indices}")
    return indices


def resolve_checkpoint(path: str) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def choose_ply_path(opt: Options, args) -> str:
    if args.ply_path is not None:
        return args.ply_path
    if args.ply_paths_json is not None:
        opt.ply_paths_json = args.ply_paths_json
    if args.ply_index is None:
        raise ValueError("Provide either --ply-path or --ply-index")
    paths = load_ply_paths(opt.ply_paths_json)
    return paths[args.ply_index]


def strip_eos(tokens: np.ndarray, eos_token_id: int) -> np.ndarray:
    eos = np.where(tokens == eos_token_id)[0]
    if len(eos) > 0:
        return tokens[: eos[0]]
    return tokens


def default_segment_mesh_dir(output_mesh: str | None) -> Path | None:
    if output_mesh is None:
        return None
    output_mesh_path = Path(output_mesh)
    return output_mesh_path.with_name(f"{output_mesh_path.stem}_segments")


def segment_mesh_path(segment_mesh_dir: Path, segment_idx: int) -> Path:
    return segment_mesh_dir / f"segment_{segment_idx:04d}.ply"


def save_tokens_file(
    output_tokens: str | None,
    pred_tokens_by_segment: dict[int, np.ndarray],
    combined_tokens: np.ndarray | None,
    combined_segment_indices: Sequence[int],
) -> None:
    if output_tokens is None:
        return

    path = Path(output_tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        arrays = {f"segment_{idx}": tokens for idx, tokens in pred_tokens_by_segment.items()}
        if combined_tokens is not None:
            arrays["combined"] = combined_tokens
        arrays["combined_segment_indices"] = np.asarray(list(combined_segment_indices), dtype=np.int64)
        np.savez_compressed(path, **arrays)
        return

    write_json(
        {
            "segments": {str(idx): tokens.tolist() for idx, tokens in pred_tokens_by_segment.items()},
            "combined_segment_indices": [int(idx) for idx in combined_segment_indices],
            "combined_tokens": None if combined_tokens is None else combined_tokens.tolist(),
        },
        path,
    )


class TokenProgressBar:
    def __init__(self, total_requested_segments: int, max_new_tokens: int, enabled: bool = True):
        self.total_requested_segments = int(total_requested_segments)
        self.total = max(1, int(total_requested_segments) * max(1, int(max_new_tokens)))
        self.enabled = bool(enabled)
        self.generated_tokens = 0
        self.start_time = time.perf_counter()
        self.bar = None
        if self.enabled:
            try:
                from tqdm.auto import tqdm

                self.bar = tqdm(
                    total=self.total,
                    desc="generating",
                    unit="tok",
                    dynamic_ncols=True,
                    leave=True,
                )
            except Exception:  # noqa: BLE001
                self.bar = None

    def update(self, tokens_this_step: int, active_batch_size: int, finished_segments: int) -> None:
        tokens_this_step = int(tokens_this_step)
        active_batch_size = int(active_batch_size)
        finished_segments = int(finished_segments)
        self.generated_tokens += tokens_this_step
        elapsed = max(time.perf_counter() - self.start_time, 1e-9)
        tok_per_sec = self.generated_tokens / elapsed

        if self.bar is not None:
            self.bar.update(tokens_this_step)
            self.bar.set_postfix(
                active_bs=active_batch_size,
                finished=f"{finished_segments}/{self.total_requested_segments}",
                tok_s=f"{tok_per_sec:.2f}",
            )
        elif self.enabled:
            print(
                f"\rgenerating: {self.generated_tokens} tok | "
                f"{tok_per_sec:.2f} tok/s | active_bs={active_batch_size} | "
                f"finished={finished_segments}/{self.total_requested_segments}",
                end="",
                flush=True,
            )

    def close(self) -> None:
        if self.bar is not None:
            # If all segments ended with EOS before the theoretical max length,
            # make the bar finish at the number of tokens that were actually decoded.
            if self.generated_tokens < self.total:
                self.bar.total = max(1, self.generated_tokens)
                self.bar.refresh()
            self.bar.close()
        elif self.enabled:
            print()


def main():
    args = parse_args()
    segment_indices = parse_segment_indices(args.segment_indices, args.segment_idx)

    ckpt_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    opt = Options.from_dict(checkpoint.get("opt", {}))
    if args.use_meto is not None:
        opt.use_meto = args.use_meto
    if args.max_new_tokens is not None:
        opt.max_new_tokens = args.max_new_tokens
    if args.generate_mode is not None:
        opt.generate_mode = args.generate_mode
    if args.temperature is not None:
        opt.temperature = args.temperature
    if args.top_k is not None:
        opt.top_k = args.top_k

    for segment_idx in segment_indices:
        if not 0 <= segment_idx < opt.num_segments:
            raise ValueError(f"segment_idx must be in [0, {opt.num_segments - 1}], got {segment_idx}")

    seed = int(args.seed if args.seed is not None else opt.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer, tokenizer_vocab_size = get_tokenizer(opt)
    vocab_size = int(checkpoint.get("vocab_size", tokenizer_vocab_size))
    opt.vocab_size = vocab_size

    device = torch.device(args.device)
    model = LMM(opt, vocab_size=vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    if args.bf16 and device.type == "cuda":
        model.to(dtype=torch.bfloat16)
    model.eval()

    ply_path = choose_ply_path(opt, args)
    coords, points, _, _ = load_tokenized_ply(ply_path, opt, tokenizer=tokenizer, training=False)
    total_tokens = torch.tensor(len(coords), dtype=torch.long, device=device)
    points_tensor = torch.from_numpy(points).to(device=device)

    max_new_tokens = opt.max_new_tokens
    if max_new_tokens is None or int(max_new_tokens) <= 0:
        max_new_tokens = max(1, int(opt.max_segment_length))
    else:
        max_new_tokens = min(int(max_new_tokens), max(1, int(opt.max_segment_length)))

    progress = TokenProgressBar(
        total_requested_segments=len(segment_indices),
        max_new_tokens=max_new_tokens,
        enabled=args.progress,
    )
    try:
        with torch.no_grad():
            pred_by_segment = model.generate_segments(
                points=points_tensor,
                total_tokens=total_tokens,
                segment_indices=segment_indices,
                max_new_tokens=opt.max_new_tokens,
                temperature=opt.temperature,
                top_k=opt.top_k,
                generate_mode=opt.generate_mode,
                progress_callback=progress.update if args.progress else None,
            )
    finally:
        progress.close()

    pred_np_by_segment = {
        int(segment_idx): pred.detach().cpu().numpy().astype(np.int64)
        for segment_idx, pred in pred_by_segment.items()
    }
    stripped_np_by_segment = {
        segment_idx: strip_eos(tokens, opt.eos_token_id)
        for segment_idx, tokens in pred_np_by_segment.items()
    }

    combined_tokens = None
    combined_segment_indices = sorted(segment_indices)
    per_segment_mesh_paths: dict[int, str] = {}

    if args.output_mesh is not None:
        segment_mesh_dir = Path(args.output_segment_mesh_dir) if args.output_segment_mesh_dir else default_segment_mesh_dir(args.output_mesh)
        assert segment_mesh_dir is not None
        for segment_idx in combined_segment_indices:
            path = segment_mesh_path(segment_mesh_dir, segment_idx)
            save_mesh_from_tokens(pred_np_by_segment[segment_idx], opt, path, tokenizer=tokenizer, clean=True)
            per_segment_mesh_paths[segment_idx] = str(path)

        if args.reconstruct_with_gt_context:
            gt_segments = split_coords_into_segments(
                coords,
                num_segments=opt.num_segments,
                boundary_token=opt.segment_boundary_token,
                force_start_token=opt.force_segment_start_token,
                strict_boundary=opt.strict_segment_boundary,
            )
            for segment_idx in segment_indices:
                gt_segments[segment_idx] = stripped_np_by_segment[segment_idx]
            combined_tokens = np.concatenate(gt_segments, axis=0)
        else:
            combined_tokens = np.concatenate(
                [stripped_np_by_segment[segment_idx] for segment_idx in combined_segment_indices],
                axis=0,
            )
        save_mesh_from_tokens(combined_tokens, opt, args.output_mesh, tokenizer=tokenizer, clean=True)

    save_tokens_file(
        args.output_tokens,
        pred_np_by_segment,
        combined_tokens,
        combined_segment_indices,
    )

    print(f"generated segments: {segment_indices}")
    for segment_idx in combined_segment_indices:
        token_count = len(pred_np_by_segment[segment_idx])
        clean_count = len(stripped_np_by_segment[segment_idx])
        eos_note = " with EOS" if token_count != clean_count else ""
        mesh_note = f"; mesh={per_segment_mesh_paths[segment_idx]}" if segment_idx in per_segment_mesh_paths else ""
        print(f"  segment {segment_idx}: {token_count} tokens{eos_note}; mesh tokens={clean_count}{mesh_note}")
    if args.output_tokens is not None:
        print(f"saved tokens to {args.output_tokens}")
    if args.output_mesh is not None:
        if args.reconstruct_with_gt_context:
            print(f"saved combined mesh with GT context to {args.output_mesh}")
        else:
            print(f"saved combined mesh sorted by segment_idx to {args.output_mesh}")


if __name__ == "__main__":
    main()
