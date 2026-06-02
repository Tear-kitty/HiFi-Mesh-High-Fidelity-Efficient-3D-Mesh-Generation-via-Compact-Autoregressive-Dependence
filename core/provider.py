from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh

from kiui.mesh_utils import clean_mesh, decimate_mesh
from core.options import Options
from core.utils import (
    load_mesh,
    load_ply_paths,
    make_autoregressive_pair,
    normalize_mesh,
    split_coords_into_segments,
    tokenize_mesh,
    detokenize_mesh,
)


@dataclass
class MeshSample:
    path: str
    points: np.ndarray
    coords: np.ndarray
    input_ids: np.ndarray
    labels: np.ndarray
    segment_idx: int
    total_tokens: int
    num_faces: int


def _sample_points(vertices: np.ndarray, faces: np.ndarray, point_num: int) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # mesh.export('./workspace/ori.ply')
    if len(mesh.faces) == 0:
        raise ValueError("cannot sample points from a mesh without faces")
    points = mesh.sample(point_num).astype(np.float32)
    # point_cloud = trimesh.PointCloud(points)
    # point_cloud.export("./workspace/points.ply")
    return points


def _maybe_scale(vertices: np.ndarray, training: bool, opt: Options) -> np.ndarray:
    if training and opt.use_scale_aug:
        bound = float(np.random.uniform(0.75, 0.95))
    else:
        bound = 0.95
    return normalize_mesh(vertices, bound=bound).astype(np.float32)


def load_tokenized_ply(
    path: str,
    opt: Options,
    tokenizer=None,
    training: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    vertices, faces = load_mesh(path)
    if training is True and opt.use_decimate_aug is True and random.random() < 0.5:
        target = int(np.random.choice([
            max(100, faces.shape[0] // 4),
            max(100, faces.shape[0] // 2),
            max(100, faces.shape[0] * 3 // 4),
            ]))
        vertices, faces = decimate_mesh(vertices, faces, target=target, verbose=False)
    vertices = _maybe_scale(vertices, training=training, opt=opt)
    
    coords = tokenize_mesh(vertices, faces, opt.discrete_bins, tokenizer=tokenizer)
    # vertices, faces = detokenize_mesh(coords, opt.discrete_bins, tokenizer=tokenizer)
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # mesh.export('./workspace/recon.ply')
    if len(coords) > opt.max_seq_length:
        raise ValueError(
            f"Token sequence is too long: {len(coords)} > max_seq_length={opt.max_seq_length}. path={path}"
        )
    points = _sample_points(vertices, faces, opt.point_num)
    return coords, points, faces, len(faces)


def build_training_sample(
    path: str,
    opt: Options,
    tokenizer=None,
    training: bool = False,
    segment_idx: Optional[int] = None,
) -> MeshSample:
    coords, points, faces, num_faces = load_tokenized_ply(path, opt, tokenizer, training=training)
    segments = split_coords_into_segments(
        coords,
        num_segments=opt.num_segments,
        boundary_token=opt.segment_boundary_token,
        force_start_token=opt.force_segment_start_token,
        strict_boundary=opt.strict_segment_boundary,
    )

    if segment_idx is None:
        segment_idx = random.randrange(opt.num_segments) if training else 0
    if not 0 <= segment_idx < opt.num_segments:
        raise ValueError(f"segment_idx must be in [0, {opt.num_segments - 1}], got {segment_idx}")

    segment = segments[segment_idx]
    if opt.force_segment_start_token and len(segment) > 0 and segment[0] != opt.segment_boundary_token:
        raise ValueError(
            f"Segment {segment_idx} of {path} does not start with token {opt.segment_boundary_token}."
        )

    input_ids, labels = make_autoregressive_pair(
        segment,
        bos_token_id=opt.bos_token_id,
        eos_token_id=opt.eos_token_id,
    )
    if opt.max_segment_length > 0 and len(input_ids) > opt.max_segment_length:
        raise ValueError(
            f"Segment sequence is too long: {len(input_ids)} > max_segment_length={opt.max_segment_length}. "
            f"path={path}, segment={segment_idx}"
        )

    return MeshSample(
        path=path,
        points=points,
        coords=coords,
        input_ids=input_ids,
        labels=labels,
        segment_idx=int(segment_idx),
        total_tokens=int(len(coords)),
        num_faces=int(num_faces),
    )


class PlyMeshDataset(Dataset):
    """Dataset backed by absolute PLY paths stored in ``ply_paths.json``."""

    def __init__(self, opt: Options, training: bool, tokenizer=None):
        self.opt = opt
        self.training = training
        self.tokenizer = tokenizer

        paths = load_ply_paths(opt.ply_paths_json)
        if len(paths) == 0:
            raise ValueError(f"No PLY paths found in {opt.ply_paths_json}")

        rng = random.Random(opt.seed)
        rng.shuffle(paths)
        split = 60
        self.paths = paths[:split]
        # split = int(round(len(paths) * opt.train_split))
        # split = min(max(split, 1), len(paths))
        # self.paths = paths[:split] if training else paths[split:]
        if not self.paths:
            self.paths = paths[-min(len(paths), 1) :]

    def __len__(self) -> int:
        multiplier = max(1, int(self.opt.repeat_per_epoch)) if self.training else 1
        return len(self.paths) * multiplier

    def __getitem__(self, idx: int) -> dict:
        # Retry a few neighbouring meshes so one malformed PLY does not kill a
        # long distributed run. If all retries fail, re-raise the last exception.
        last_error: Optional[Exception] = None
        for offset in range(8):
            path = self.paths[(idx + offset) % len(self.paths)]
            try:
                sample = build_training_sample(
                    path,
                    self.opt,
                    tokenizer=self.tokenizer,
                    training=self.training,
                    segment_idx=None if self.training else 0,
                )
                return {
                    "path": sample.path,
                    "points": sample.points,
                    "coords": sample.coords,
                    "input_ids": sample.input_ids,
                    "labels": sample.labels,
                    "segment_idx": sample.segment_idx,
                    "total_tokens": sample.total_tokens,
                    "num_faces": sample.num_faces,
                }
            except Exception as exc:  # noqa: BLE001 - dataset retry path
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load a valid sample near index {idx}: {last_error}")


def collate_fn(batch: list[dict], opt: Options) -> dict:
    batch_size = len(batch)
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = np.full((batch_size, max_len), opt.pad_token_id, dtype=np.int64)
    labels = np.full((batch_size, max_len), -100, dtype=np.int64)
    token_attention_mask = np.zeros((batch_size, max_len), dtype=bool)

    for row, item in enumerate(batch):
        cur_len = len(item["input_ids"])
        input_ids[row, :cur_len] = item["input_ids"]
        labels[row, :cur_len] = item["labels"]
        token_attention_mask[row, :cur_len] = True

    # The BOS input should predict the first topology token. With the requested
    # boundary strategy this target should be 5 for every segment.
    first_targets = labels[:, 0]
    if opt.force_segment_start_token and not np.all(first_targets == opt.segment_boundary_token):
        bad = np.where(first_targets != opt.segment_boundary_token)[0][0]
        raise ValueError(
            f"Autoregressive alignment error: labels[{bad}, 0]={first_targets[bad]}, "
            f"expected {opt.segment_boundary_token}."
        )

    return {
        "points": torch.from_numpy(np.stack([item["points"] for item in batch], axis=0)).float(),
        "input_ids": torch.from_numpy(input_ids).long(),
        "labels": torch.from_numpy(labels).long(),
        "token_attention_mask": torch.from_numpy(token_attention_mask).bool(),
        "segment_idx": torch.tensor([item["segment_idx"] for item in batch], dtype=torch.long),
        "total_tokens": torch.tensor([item["total_tokens"] for item in batch], dtype=torch.long),
        "num_faces": torch.tensor([item["num_faces"] for item in batch], dtype=torch.long),
        "paths": [item["path"] for item in batch],
        "coords": [item["coords"] for item in batch],
    }
