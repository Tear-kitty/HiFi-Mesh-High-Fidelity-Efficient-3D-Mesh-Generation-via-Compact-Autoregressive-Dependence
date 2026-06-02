from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh

from core.options import Options


def init_logger(filename: str, name: str = "hifi_mesh") -> logging.Logger:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(filename, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def read_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_ply_paths(path: str | os.PathLike) -> list[str]:
    data = read_json(path)
    if isinstance(data, dict):
        paths = list(data.values())
    elif isinstance(data, list):
        paths = data
    else:
        raise TypeError(f"{path} must contain a list or dict of ply paths")

    paths = [str(p) for p in paths]
    bad = [p for p in paths if not os.path.isabs(p)]
    if bad:
        raise ValueError(
            f"ply_paths_json must store absolute paths. First relative path: {bad[0]}"
        )
    return paths


def load_mesh(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    mesh_or_scene = trimesh.load(str(path), process=False)
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = []
        graph = mesh_or_scene.graph.to_flattened()
        for _, node in graph.items():
            geom_name = node.get("geometry")
            if geom_name in mesh_or_scene.geometry:
                geom = mesh_or_scene.geometry[geom_name]
                if isinstance(geom, trimesh.Trimesh):
                    geom = geom.copy()
                    geom.apply_transform(node.get("transform", np.eye(4)))
                    meshes.append(geom)
        if not meshes:
            raise ValueError(f"No trimesh geometry found in scene: {path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(mesh_or_scene, trimesh.Trimesh):
        mesh = mesh_or_scene
    else:
        raise TypeError(f"Unsupported mesh type from {path}: {type(mesh_or_scene)!r}")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Empty mesh: {path}")
    return vertices, faces


def normalize_mesh(vertices: np.ndarray, bound: float = 0.95) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmax + vmin) * 0.5
    scale = float(np.max(vmax - vmin))
    if scale <= 1e-12:
        return vertices - center
    return (vertices - center) * (2.0 * bound / scale)


def _import_meto_engine():
    """Import EdgeRunner's local meto.Engine robustly.

    EdgeRunner's source tree has a nested layout:

        meto/                 # extension project, contains setup.py
          setup.py
          meto/__init__.py    # real Python package exposing Engine
          src/bindings.cpp    # builds the _meto extension

    When training is launched from the HiFi-Mesh project root, Python can first
    see the outer ./meto directory as a namespace package and cache it in
    sys.modules as `meto` with __file__ == None. In that state,
    `from meto import Engine` fails with "unknown location" even if ./meto is
    present. We put ./meto itself on sys.path, clear stale namespace entries,
    and then import the real inner package.
    """
    import importlib
    import sys

    project_root = Path(__file__).resolve().parents[1]
    local_meto = project_root / "meto"
    inner_init = local_meto / "meto" / "__init__.py"

    if inner_init.exists():
        local_meto_str = str(local_meto)
        sys.path = [p for p in sys.path if p != local_meto_str]
        sys.path.insert(0, local_meto_str)

    for name in list(sys.modules):
        if name == "meto" or name.startswith("meto."):
            del sys.modules[name]

    try:
        module = importlib.import_module("meto")
    except Exception as exc:
        raise ImportError(
            "Could not import EdgeRunner's meto package. Expected files: "
            "./meto/setup.py, ./meto/meto/__init__.py, ./meto/src/bindings.cpp. "
            "Install it with: python -m pip install pybind11 && "
            "python -m pip install --no-build-isolation -e ./meto. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    Engine = getattr(module, "Engine", None)
    if Engine is None:
        raise ImportError(
            "Imported a module named 'meto', but it does not expose Engine. "
            f"meto.__file__={getattr(module, '__file__', None)!r}, "
            f"meto.__path__={list(getattr(module, '__path__', []))!r}. "
            "This usually means Python imported the outer ./meto namespace folder "
            "instead of ./meto/meto/__init__.py, or meto was copied with the wrong "
            "directory nesting. Expected nesting: ./meto/meto/__init__.py."
        )
    return Engine


def get_tokenizer(opt: Options):
    if opt.use_meto:
        try:
            Engine = _import_meto_engine()
        except Exception as exc:
            raise ImportError(
                "opt.use_meto=True requires EdgeRunner's local `./meto` tokenizer. "
                "Copy the original NVlabs/EdgeRunner/meto folder to the project root, then run:\n"
                "  python -m pip install pybind11 numpy trimesh kiui pymeshlab\n"
                "  python -m pip install --no-build-isolation -e ./meto\n"
                "If import still fails from the project root, run with:\n"
                "  PYTHONPATH=$PWD/meto:$PWD accelerate launch ...\n"
                "For pipeline debugging only, launch with --use-meto false."
            ) from exc
        tokenizer = Engine(discrete_bins=opt.discrete_bins, backend=opt.meto_backend)
        vocab_size = int(tokenizer.num_tokens) + 3
    else:
        tokenizer = None
        vocab_size = int(opt.discrete_bins) + 3
    return tokenizer, vocab_size

def tokenize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    discrete_bins: int,
    tokenizer=None,
) -> np.ndarray:
    """Convert a mesh to token ids. Special ids 0/1/2 are reserved."""
    if tokenizer is not None:
        tokens, _, _ = tokenizer.encode(vertices, faces)
        return np.asarray(tokens, dtype=np.int64) + 3

    sort_inds = np.lexsort(vertices.T)
    vertices = vertices[sort_inds][:, [2, 1, 0]]
    inv_inds = np.argsort(sort_inds)
    faces = inv_inds[faces]

    start_inds = faces.argmin(axis=1)
    all_inds = start_inds[:, None] + np.arange(3)[None, :]
    faces_5 = np.concatenate([faces, faces[:, :2]], axis=1)
    faces = np.take_along_axis(faces_5, all_inds, axis=1)
    faces = np.array(sorted(faces.tolist()), dtype=np.int64)

    verts_per_face = vertices[faces]
    coords = ((verts_per_face + 1.0) * 0.5 * discrete_bins)
    coords = coords.clip(0, discrete_bins - 1).astype(np.int64)
    return coords.reshape(-1) + 3


def detokenize_mesh(tokens: Sequence[int] | np.ndarray, discrete_bins: int, tokenizer=None):
    tokens = np.asarray(tokens, dtype=np.int64)
    tokens = tokens[(tokens != 0) & (tokens != 1) & (tokens != 2)]
    if len(tokens) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)
    tokens = tokens - 3
    if tokenizer is not None:
        vertices, faces, _ = tokenizer.decode(tokens)
        return vertices, faces

    if len(tokens) % 9 != 0:
        tokens = tokens[: len(tokens) - (len(tokens) % 9)]
    if len(tokens) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    invalid = (tokens < 0).reshape(-1, 9).any(axis=1)
    coords = tokens.reshape(-1, 3)
    vertices = (coords.astype(np.float32) + 0.5) / discrete_bins * 2.0 - 1.0
    vertices = vertices[:, [2, 1, 0]]
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    faces = faces[~invalid]
    return vertices, faces


def save_mesh_from_tokens(
    tokens: Sequence[int] | np.ndarray,
    opt: Options,
    path: str | os.PathLike,
    tokenizer=None,
    clean: bool = True,
) -> None:
    tokens = np.asarray(tokens, dtype=np.int64)
    eos = np.where(tokens == opt.eos_token_id)[0]
    if len(eos) > 0:
        tokens = tokens[: eos[0]]
    vertices, faces = detokenize_mesh(tokens, opt.discrete_bins, tokenizer=tokenizer)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if clean and len(mesh.faces) > 0:
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def split_coords_into_segments(
    coords: Sequence[int] | np.ndarray,
    num_segments: int = 10,
    boundary_token: int = 5,
    force_start_token: bool = True,
    strict_boundary: bool = False,
) -> list[np.ndarray]:
    """Split a tokenized mesh sequence into equal-length segments.

    For the 9 internal boundaries, the target location is the exact 10-way
    equal split. The actual boundary is moved to the nearest token whose value
    is ``boundary_token``. The boundary token itself belongs to the next segment,
    therefore every non-first boundary segment starts with that token.
    """
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 1:
        raise ValueError("coords must be a 1-D token sequence")
    if len(coords) == 0:
        raise ValueError("coords is empty")

    token_positions = np.flatnonzero(coords == boundary_token)
    boundaries = [0]
    n = len(coords)

    for seg_idx in range(1, num_segments):
        target = int(round(n * seg_idx / num_segments))
        min_allowed = boundaries[-1] + 1
        max_allowed = n - (num_segments - seg_idx)
        max_allowed = max(min_allowed, max_allowed)

        candidates = token_positions[
            (token_positions >= min_allowed) & (token_positions <= max_allowed)
        ]
        if len(candidates) > 0:
            boundary = int(candidates[np.argmin(np.abs(candidates - target))])
        elif strict_boundary:
            raise ValueError(
                f"Cannot find boundary token {boundary_token} near split {seg_idx}/{num_segments}."
            )
        else:
            boundary = int(np.clip(target, min_allowed, max_allowed))
        boundaries.append(boundary)

    boundaries.append(n)
    segments: list[np.ndarray] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        segment = coords[left:right]
        if len(segment) == 0:
            if strict_boundary:
                raise ValueError("empty segment after boundary search")
            segment = np.asarray([boundary_token], dtype=np.int64)

        if force_start_token and segment[0] != boundary_token:
            start_candidates = np.flatnonzero(segment == boundary_token)
            if len(start_candidates) > 0:
                segment = segment[int(start_candidates[0]) :]
            elif strict_boundary:
                raise ValueError(
                    f"Segment does not start with boundary token {boundary_token} and no token can be used to repair it."
                )
        segments.append(segment.astype(np.int64, copy=False))
    return segments


def make_autoregressive_pair(
    segment_tokens: Sequence[int] | np.ndarray,
    bos_token_id: int,
    eos_token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    segment_tokens = np.asarray(segment_tokens, dtype=np.int64)
    seq = np.concatenate(
        [
            np.asarray([bos_token_id], dtype=np.int64),
            segment_tokens,
            np.asarray([eos_token_id], dtype=np.int64),
        ]
    )
    return seq[:-1], seq[1:]


def cosine_schedule_with_warmup(
    current_step: int,
    total_steps: int,
    warmup_ratio: float = 0.0,
    min_ratio: float = 0.1,
) -> float:
    if total_steps <= 0:
        return 1.0
    progress = current_step / float(max(1, total_steps))
    if warmup_ratio > 0 and progress < warmup_ratio:
        return progress / warmup_ratio
    denom = max(1e-8, 1.0 - warmup_ratio)
    progress = min(1.0, max(0.0, (progress - warmup_ratio) / denom))
    return max(min_ratio, min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
