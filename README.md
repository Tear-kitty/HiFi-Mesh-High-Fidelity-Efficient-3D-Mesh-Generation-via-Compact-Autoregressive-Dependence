# HiFi-Mesh: High-Fidelity-Efficient-3D-Mesh-Generation-via-Compact-Autoregressive-Dependence
<p align="center">
  <a href="https://arxiv.org/abs/2601.21314"><img src="https://img.shields.io/badge/arXiv-2601.21314-b31b1b.svg" alt="arXiv"></a>
  <a href="https://doi.org/10.1609/aaai.v40i8.37586"><img src="https://img.shields.io/badge/AAAI-2026-blue.svg" alt="AAAI 2026"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-see%20LICENSE-lightgrey.svg" alt="License"></a>
</p>

Official implementation of **HiFi-Mesh: High-Fidelity Efficient 3D Mesh Generation via Compact Autoregressive Dependence**.

HiFi-Mesh is a high-fidelity and efficient autoregressive 3D mesh generation framework. It tokenizes meshes into one-dimensional sequences and learns compact autoregressive dependencies for segment-wise mesh generation. The method introduces **Latent Autoregressive Network (LANE)** for compact dependencies and **Adaptive Computation Graph Reconfiguration (AdaGraph)** for efficient inference.

> **Paper:** HiFi-Mesh: High-Fidelity Efficient 3D Mesh Generation via Compact Autoregressive Dependence  
> **Venue:** AAAI 2026  
> **arXiv:** https://arxiv.org/abs/2601.21314

<p align="center">
  <!-- Replace this with your teaser figure or demo gif. -->
  <img src="assets/teaser.png" width="90%" alt="HiFi-Mesh teaser">
</p>

## News

- **2026-03-14:** HiFi-Mesh was published in the Proceedings of the AAAI Conference on Artificial Intelligence.
- **2026-01-29:** The arXiv preprint was released.

## Project Structure

```text
HiFi-Mesh/
├── acc_configs/              # HuggingFace Accelerate configs
├── core/
│   ├── models.py             # LMM / LANE model implementation
│   ├── options.py            # Default configuration
│   ├── provider.py           # PLY dataset and dataloader
│   ├── utils.py              # Mesh I/O, tokenization, segmentation helpers
│   └── transformer/          # Transformer blocks and attention modules
├── dataset/
│   └── ply_paths.json        # Example path list file
├── meto/                     # EdgeRunner mesh tokenizer, installed locally
├── scripts/
│   ├── install_meto.sh
│   ├── diagnose_meto.py
│   └── debug_meto.py
├── main.py                   # Training entry point
├── infer.py                  # Segment and multi-segment inference entry point
└── requirements.txt
```

## Installation

The environment follows the EdgeRunner-style setup: install CUDA-enabled PyTorch first, then install `flash-attn`, the local `meto` mesh tokenizer, and the remaining Python dependencies.

This repository was tested with the following setup:

```text
Ubuntu 22.04
Python 3.10
CUDA 12.4
PyTorch 2.4.1 + cu124
flash-attn 2.8.3
```

Other CUDA versions may also work, but PyTorch, CUDA, and `flash-attn` should be kept compatible.

### 1. Clone the repository

```bash
git clone https://github.com/<your-name>/HiFi-Mesh.git
cd HiFi-Mesh
```

### 2. Create a conda environment

```bash
conda create -n hifimesh python=3.10 -y
conda activate hifimesh

python -m pip install -U pip wheel
python -m pip install "setuptools==69.5.1" ninja packaging
```

### 3. Install PyTorch

For CUDA 12.4:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

For CUDA 12.1:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Check the installation:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

### 4. Install common dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Install flash-attn

For training, `flash-attn` is recommended. Install it after PyTorch is installed:

```bash
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

If `flash-attn` compilation fails, first check that your CUDA toolkit, PyTorch CUDA version, compiler, and GPU architecture are compatible. Inference can fall back to ordinary attention in the current implementation, but training is expected to be much faster with `flash-attn`.

### 6. Install the EdgeRunner `meto` mesh tokenizer

HiFi-Mesh uses the EdgeRunner-style `meto` tokenizer. Do **not** install an unrelated PyPI package named `meto`. The expected directory layout is:

```text
HiFi-Mesh/
└── meto/
    ├── setup.py
    ├── meto/
    │   └── __init__.py
    ├── src/
    │   └── bindings.cpp
    └── include/
```

If your repository does not already contain the full tokenizer source, copy it from EdgeRunner:

```bash
git clone --depth 1 https://github.com/NVlabs/EdgeRunner.git /tmp/EdgeRunner
rm -rf ./meto
cp -r /tmp/EdgeRunner/meto ./meto
```

Then build and install it:

```bash
bash scripts/install_meto.sh
```

Manual installation is also possible:

```bash
python -m pip install pybind11
python -m pip install --no-build-isolation -e ./meto
```

Verify `meto`:

```bash
PYTHONPATH=$PWD/meto:$PWD python scripts/diagnose_meto.py
```

For dataloader or model debugging only, you may run with `--use-meto false`. This uses a fallback coordinate tokenizer and is not recommended for final training if you want EdgeRunner-style topology-token behavior.

## Data Preparation

`dataset/ply_paths.json` should contain either a JSON list of absolute `.ply` file paths or a dictionary whose values are absolute `.ply` file paths.

Example list format:

```json
[
  "/absolute/path/to/mesh_000001.ply",
  "/absolute/path/to/mesh_000002.ply"
]
```

## Training

Before training, make sure the GPU selection is controlled by your shell or job scheduler. For open-source use, avoid hard-coding a single GPU ID inside `main.py`.

### Single-GPU debug training

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file acc_configs/gpu1.yaml main.py \
  --ply-paths-json dataset/ply_paths.json \
  --workspace ./workspace/debug \
  --log-dir ./log/debug \
  --resume "" \
  --start-epoch 0 \
  --batch-size 1 \
  --num-workers 4 \
  --segment-latent-tokens 64 \
  --use-meto true
```

### Multi-GPU training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file acc_configs/gpu8.yaml main.py \
  --ply-paths-json dataset/ply_paths.json \
  --workspace ./workspace/hifimesh \
  --log-dir ./log/hifimesh \
  --resume "" \
  --start-epoch 0 \
  --batch-size 6 \
  --num-workers 8 \
  --segment-latent-tokens 64 \
  --mixed-precision bf16 \
  --use-meto true
```

Checkpoints are saved as:

```text
workspace/hifimesh/checkpoint-0009/checkpoint.pth
workspace/hifimesh/checkpoint-0019/checkpoint.pth
...
```

The main logged metrics are:

```text
loss
 token_acc
 first_token_acc
 mean_segment_idx
 mean_segment_len
```

To resume training:

```bash
accelerate launch --config_file acc_configs/gpu8.yaml main.py \
  --ply-paths-json dataset/ply_paths.json \
  --workspace ./workspace/hifimesh \
  --resume ./workspace/hifimesh/checkpoint-0199 \
  --start-epoch 199
```

If `--resume` points to a directory, the code will look for `checkpoint.pth` inside that directory.

## Inference

`infer.py` supports both single-segment and multi-segment autoregressive inference. A checkpoint can be either a `.pth` file or a checkpoint directory containing `checkpoint.pth`.

### Single-segment inference

```bash
python infer.py \
  --checkpoint ./workspace/hifimesh/checkpoint-0199 \
  --ply-path /absolute/path/to/input_mesh.ply \
  --segment-idx 3 \
  --generate-mode greedy \
  --output-tokens ./results/segment_0003_tokens.json \
  --output-mesh ./results/segment_0003.ply
```

### Batched multi-segment inference

You can decode multiple segments synchronously by passing an index array:

```bash
python infer.py \
  --checkpoint ./workspace/hifimesh/checkpoint-0199 \
  --ply-path /absolute/path/to/input_mesh.ply \
  --segment-indices "[0,1,3,4]" \
  --generate-mode sample \
  --temperature 0.8 \
  --top-k 5 \
  --output-tokens ./results/generated_tokens.json \
  --output-mesh ./results/generated_combined.ply \
  --output-segment-mesh-dir ./results/generated_segments \
  --progress true
```

The following `--segment-indices` formats are supported:

```bash
--segment-indices "[0,1,3,4]"
--segment-indices "0,1,3,4"
--segment-indices 0 1 3 4
```

At least one segment index is required.

During batched autoregressive inference, all active segments are decoded in parallel. Once a segment predicts `EOS`, that segment is removed from the active batch. Therefore, if four segments are active at a decoding step, the progress bar counts that step as four generated tokens. If one segment finishes, the next decoding step continues with three active segments.

The inference script saves:

```text
results/generated_segments/segment_0000.ply
results/generated_segments/segment_0001.ply
results/generated_segments/segment_0003.ply
results/generated_segments/segment_0004.ply
results/generated_combined.ply
results/generated_tokens.json
```

The combined mesh is concatenated in ascending `segment_idx` order. For example, if the input order is `[4,0,3,1]`, the combined output order is `[0,1,3,4]`.

### Reconstruct with ground-truth context

For debugging, you can replace selected GT segments with generated segments while keeping the other GT segments unchanged:

```bash
python infer.py \
  --checkpoint ./workspace/hifimesh/checkpoint-0199 \
  --ply-path /absolute/path/to/input_mesh.ply \
  --segment-indices "[0,1,3,4]" \
  --reconstruct-with-gt-context true \
  --output-mesh ./results/reconstructed_with_gt_context.ply
```

This is useful for checking whether the generated segment is geometrically compatible with the rest of the original mesh.

## Useful Arguments

### Training

| Argument | Default | Description |
| --- | ---: | --- |
| `--ply-paths-json` | `dataset/ply_paths.json` | JSON file containing absolute PLY paths. |
| `--workspace` | `./workspace` | Directory for checkpoints. |
| `--log-dir` | `./log` | Directory for training logs. |
| `--resume` | `""` recommended for new training | Checkpoint file or checkpoint directory. |
| `--num-segments` | `10` | Number of mesh segments. |
| `--segment-latent-tokens` | `64` | Number of latent tokens per segment. |
| `--hidden-dim` | `1536` | Transformer hidden dimension. |
| `--num-layers` | `24` | Number of decoder layers. |
| `--point-num` | `131072` | Number of sampled point-cloud condition points. |
| `--max-seq-length` | `200000` | Maximum full mesh token length. |
| `--max-segment-length` | `25000` | Maximum target segment token length. |
| `--batch-size` | `6` | Per-process batch size. |
| `--mixed-precision` | `bf16` | Accelerate mixed precision mode. |
| `--use-meto` | `true` | Use EdgeRunner `meto` tokenizer. |

### Inference

| Argument | Description |
| --- | --- |
| `--checkpoint` | Checkpoint `.pth` file or checkpoint directory. |
| `--ply-path` | Conditioning PLY mesh path. |
| `--ply-index` | Optional index into `ply_paths_json`. |
| `--segment-idx` | Single segment index, used when `--segment-indices` is omitted. |
| `--segment-indices` | One or more segment indices for batched decoding. |
| `--generate-mode` | `greedy` or `sample`. |
| `--temperature` | Sampling temperature. |
| `--top-k` | Sampling top-k argument. |
| `--output-tokens` | Save generated tokens as `.json` or `.npz`. |
| `--output-mesh` | Save combined mesh. |
| `--output-segment-mesh-dir` | Save each generated segment as an individual mesh. |
| `--reconstruct-with-gt-context` | Replace selected GT segments and keep the rest of GT context. |
| `--progress` | Show token/s progress during autoregressive decoding. |

### Out of memory

Try reducing:

```text
--batch-size
--point-num
--hidden-dim
--num-layers
--point-latent-size-out
--max-segment-length
```

For quick debugging, start with `--batch-size 1` and `--point-num 32768`.

## Acknowledgements

This project builds on ideas and open-source components from the 3D mesh generation community. We especially thank:

- [EdgeRunner](https://github.com/NVlabs/EdgeRunner) for the mesh tokenizer and related autoregressive mesh generation infrastructure.
- [flash-attention](https://github.com/Dao-AILab/flash-attention) for efficient attention kernels.
- [HuggingFace Accelerate](https://github.com/huggingface/accelerate) for distributed training utilities.
- [trimesh](https://github.com/mikedh/trimesh) for mesh processing utilities.

## Citation

If you find this project useful, please cite:

```bibtex
@article{li2026hifimesh,
  title   = {HiFi-Mesh: High-Fidelity Efficient 3D Mesh Generation via Compact Autoregressive Dependence},
  author  = {Li, Yanfeng and Tan, Tao and Gao, Qinquan and Cao, Zhiwen and Liu, Xiaohong and Sun, Yue},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {8},
  pages   = {6566--6574},
  year    = {2026},
  doi     = {10.1609/aaai.v40i8.37586}
}
```

## License

Please see [LICENSE](./LICENSE) for details. The `meto` tokenizer follows the license of the original EdgeRunner release.
