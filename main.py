from __future__ import annotations

import argparse
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from core.models import LMM
from core.options import Options
from core.provider import PlyMeshDataset, collate_fn
from core.utils import cosine_schedule_with_warmup, get_tokenizer, init_logger


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def parse_args() -> Options:
    defaults = Options()
    parser = argparse.ArgumentParser(description="HiFi-Mesh hierarchical autoregressive training")

    # Frequently changed experiment knobs are intentionally exposed here.
    parser.add_argument("--ply-paths-json", default=defaults.ply_paths_json)
    parser.add_argument("--workspace", default=defaults.workspace)
    parser.add_argument("--log-dir", default=defaults.log_dir)
    parser.add_argument("--resume", default=defaults.resume)
    parser.add_argument("--start-epoch", type=int, default=defaults.start_epoch)

    parser.add_argument("--segment-latent-tokens", type=int, default=defaults.segment_latent_tokens)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--point-encoder-dim", type=int, default=defaults.point_encoder_dim)
    parser.add_argument("--point-encoder-dim-mid", type=int, default=defaults.point_encoder_dim_mid)
    parser.add_argument("--point-encoder-dim-out", type=int, default=defaults.point_encoder_dim_out)
    parser.add_argument("--point-encoder-heads", type=int, default=defaults.point_encoder_heads)
    parser.add_argument("--point-encoder-heads-mid", type=int, default=defaults.point_encoder_heads_mid)
    parser.add_argument("--point-encoder-heads-out", type=int, default=defaults.point_encoder_heads_out)
    parser.add_argument("--point-latent-size", type=int, default=defaults.point_latent_size)
    parser.add_argument("--point-latent-size-mid", type=int, default=defaults.point_latent_size_mid)
    parser.add_argument("--point-latent-size-out", type=int, default=defaults.point_latent_size_out)
    parser.add_argument("--point-num", type=int, default=defaults.point_num)

    parser.add_argument("--num-segments", type=int, default=defaults.num_segments)
    parser.add_argument("--segment-boundary-token", type=int, default=defaults.segment_boundary_token)
    parser.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    parser.add_argument("--max-segment-length", type=int, default=defaults.max_segment_length)
    parser.add_argument("--strict-segment-boundary", type=str2bool, default=defaults.strict_segment_boundary)

    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--min-lr-ratio", type=float, default=defaults.min_lr_ratio)
    parser.add_argument("--gradient-clip", type=float, default=defaults.gradient_clip)
    parser.add_argument("--mixed-precision", default=defaults.mixed_precision)
    parser.add_argument("--checkpointing", type=str2bool, default=defaults.checkpointing)
    parser.add_argument("--save-checkpoint", type=str2bool, default=defaults.save_checkpoint)
    parser.add_argument("--checkpointing-epoch", type=int, default=defaults.checkpointing_epoch)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-batches", type=int, default=defaults.eval_batches)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--repeat-per-epoch", type=int, default=defaults.repeat_per_epoch)
    parser.add_argument("--train-split", type=float, default=defaults.train_split)
    parser.add_argument("--use-meto", type=str2bool, default=defaults.use_meto)
    parser.add_argument("--meto-backend", default=defaults.meto_backend)
    parser.add_argument("--discrete-bins", type=int, default=defaults.discrete_bins)
    parser.add_argument("--use-wandb", type=str2bool, default=defaults.use_wandb)

    args = parser.parse_args()
    opt = Options()
    for key, value in vars(args).items():
        setattr(opt, key.replace("-", "_"), value)
    return opt


def build_scheduler(optimizer, opt: Options, total_steps: int):
    def lr_lambda(step: int) -> float:
        return cosine_schedule_with_warmup(
            step,
            total_steps=total_steps,
            warmup_ratio=opt.warmup_ratio,
            min_ratio=opt.min_lr_ratio,
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def resolve_checkpoint_path(path: str | os.PathLike) -> Path:
    path = Path(path)
    if path.is_dir():
        candidate = path / "checkpoint.pth"
        if candidate.exists():
            return candidate
    return path


def save_checkpoint(accelerator: Accelerator, model, optimizer, scheduler, opt: Options, epoch: int, global_step: int, vocab_size: int):
    if not accelerator.is_main_process:
        return
    save_dir = Path(opt.workspace) / f"checkpoint-{epoch:04d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    state = {
        "model_state_dict": unwrapped.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "opt": opt.to_dict(),
        "vocab_size": vocab_size,
    }
    accelerator.save(state, save_dir / "checkpoint.pth")


def load_checkpoint(accelerator: Accelerator, model, optimizer, scheduler, opt: Options, logger) -> tuple[int, int]:
    if not opt.resume:
        return opt.start_epoch, 0
    ckpt_path = resolve_checkpoint_path(opt.resume)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    unwrapped = accelerator.unwrap_model(model)
    missing, unexpected = unwrapped.load_state_dict(checkpoint["model_state_dict"], strict=False)
    logger.info("Loaded checkpoint: %s", ckpt_path)
    if missing:
        logger.info("Missing keys: %s", missing)
    if unexpected:
        logger.info("Unexpected keys: %s", unexpected)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", opt.start_epoch)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    return start_epoch, global_step


def reduce_stats(accelerator: Accelerator, stats: dict) -> dict:
    reduced = {}
    for key, value in stats.items():
        if torch.is_tensor(value):
            reduced[key] = accelerator.reduce(value.detach(), reduction="mean").item()
        else:
            reduced[key] = value
    return reduced


@torch.no_grad()
def evaluate(model, dataloader, accelerator: Accelerator, max_batches: int) -> dict:
    model.eval()
    totals = {"loss": 0.0, "token_acc": 0.0, "first_token_acc": 0.0}
    count = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        _, stats = model(batch)
        reduced = reduce_stats(accelerator, stats)
        for key in totals:
            totals[key] += float(reduced[key])
        count += 1
    model.train()
    if count == 0:
        return totals
    return {key: value / count for key, value in totals.items()}


def main():
    opt = parse_args()
    set_seed(opt.seed)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    Path(opt.workspace).mkdir(parents=True, exist_ok=True)
    Path(opt.log_dir).mkdir(parents=True, exist_ok=True)
    logger = init_logger(Path(opt.log_dir) / f"train_rank{accelerator.process_index}.log")

    tokenizer, vocab_size = get_tokenizer(opt)
    opt.vocab_size = vocab_size

    train_dataset = PlyMeshDataset(opt, training=True, tokenizer=tokenizer)
    val_dataset = PlyMeshDataset(opt, training=False, tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn, opt=opt),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.eval_batch,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, opt=opt),
    )

    model = LMM(opt, vocab_size=vocab_size)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info("Options: %s", opt)
        logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))
        logger.info("Params: trainable %.3fM / total %.3fM", trainable / 1e6, total / 1e6)
        logger.info("Segment latent tokens per segment: %d", opt.segment_latent_tokens)
        logger.info(
            "Point encoder stages: tokens=%s dims=%s heads=%s",
            [opt.point_latent_size, opt.point_latent_size_mid, opt.point_latent_size_out],
            [opt.point_encoder_dim, opt.point_encoder_dim_mid, opt.point_encoder_dim_out],
            [opt.point_encoder_heads, opt.point_encoder_heads_mid, opt.point_encoder_heads_out],
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        betas=(0.9, 0.95),
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, opt.gradient_accumulation_steps)))
    total_steps = max(1, opt.num_epochs * steps_per_epoch)
    scheduler = build_scheduler(optimizer, opt, total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
    )

    if opt.use_wandb and accelerator.is_main_process:
        import wandb

        wandb.init(project="hifi-mesh", config=asdict(opt), name=Path(opt.workspace).name)

    start_epoch, global_step = load_checkpoint(accelerator, model, optimizer, scheduler, opt, logger)

    for epoch in range(start_epoch, opt.num_epochs):
        model.train()
        epoch_start = time.time()
        running = {"loss": 0.0, "token_acc": 0.0, "first_token_acc": 0.0}
        num_logs = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                loss, stats = model(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients and opt.gradient_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
                optimizer.step()
                scheduler.step()

            reduced = reduce_stats(accelerator, stats)
            for key in running:
                running[key] += float(reduced[key])
            num_logs += 1
            global_step += 1

            if accelerator.is_main_process and global_step % 1 == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "epoch=%d step=%d global_step=%d loss=%.4f token_acc=%.4f first_acc=%.4f lr=%.3e seg=%.2f len=%.1f",
                    epoch,
                    step,
                    global_step,
                    reduced["loss"],
                    reduced["token_acc"],
                    reduced["first_token_acc"],
                    lr,
                    reduced["mean_segment_idx"],
                    reduced["mean_segment_len"],
                )
                if opt.use_wandb:
                    import wandb

                    wandb.log({f"train/{k}": v for k, v in reduced.items()} | {"lr": lr, "epoch": epoch}, step=global_step)

        accelerator.wait_for_everyone()
        avg = {key: value / max(1, num_logs) for key, value in running.items()}
        if accelerator.is_main_process:
            logger.info(
                "epoch=%d done in %.1fs | train_loss=%.4f token_acc=%.4f first_acc=%.4f",
                epoch,
                time.time() - epoch_start,
                avg["loss"],
                avg["token_acc"],
                avg["first_token_acc"],
            )

        if opt.save_checkpoint and (epoch + 1) % opt.checkpointing_epoch == 0:
            accelerator.wait_for_everyone()
            save_checkpoint(accelerator, model, optimizer, scheduler, opt, epoch, global_step, vocab_size)
            
        # if opt.eval_every > 0 and (epoch + 1) % opt.eval_every == 0:
        #     val_stats = evaluate(model, val_loader, accelerator, opt.eval_batches)
        #     if accelerator.is_main_process:
        #         logger.info("epoch=%d val: %s", epoch, val_stats)
        #         if opt.use_wandb:
        #             import wandb

        #             wandb.log({f"val/{k}": v for k, v in val_stats.items()}, step=global_step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(accelerator, model, optimizer, scheduler, opt, opt.num_epochs - 1, global_step, vocab_size)
        logger.info("training finished")


if __name__ == "__main__":
    main()
