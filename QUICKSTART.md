# TeleBoost Quickstart

This walks through (1) building the image, (2) running a smoke test on
FakeDataset, (3) bringing up real DPO training, and (4) writing your
own dataset adapter when `teleai_data_tool` is not available.

---

## 0. Prerequisites

* **Hardware**: 8×NVIDIA H100 / H200 / H800 (SM 9.0) for the headline
  config. SM 8.0 GPUs (A100, etc.) work with `--build-arg BUILD_FA3=0`.
* **Driver**: CUDA 13.0-compatible NVIDIA driver (≥575.x recommended).
* **Disk**: 200 GB free for the image; more for checkpoints.
* **RAM**: 256 GB+ host memory recommended for distributed_vae mode.

---

## 1. Build the image

```bash
git clone https://github.com/Tele-AI/TeleBoost.git
cd TeleBoost

# Hopper (H100/H200/H800):
docker build -t teleboost:mc0.16.1 .
# ~80 min: pip deps (5m) + flash-attn 2 source build (35m) + flash-attn 3 source build (40m)

# Non-Hopper (skip flash-attn 3, ~35 min total):
docker build --build-arg BUILD_FA3=0 -t teleboost:mc0.16.1 .

# Behind GFW, set a pip mirror for the python deps stage:
docker build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t teleboost:mc0.16.1 .
```

What's inside (verified ABI-aligned):

| Component | Version | Source |
|---|---|---|
| Base image | `nvcr.io/nvidia/pytorch:25.09-py3` | NGC |
| torch | 2.9.0a0+nv25.09 | NGC bundled |
| CUDA | 13.0 | NGC bundled |
| transformer_engine | 2.7.0 | NGC bundled |
| apex (FusedAdam, etc.) | NGC build | NGC bundled |
| flash-attn 2 | 2.8.3 | source build |
| flash-attn 3 (Hopper) | 2.8.3 | source build |
| megatron-core | 0.16.1 | pip |
| deepspeed | **0.17.5** (pinned) | pip |
| All other pythons | see `requirements.txt` | pip |

> **Why deepspeed 0.17.5 specifically?** Versions 0.17.6+ replaced the
> simple multi-call epilogue with an `all_grad_tensors` state machine
> that requires `DeepSpeedEngine` to drive `is_gradient_accumulation_boundary`.
> teletron uses `DeepSpeedZeroOptimizer` directly (no Engine wrapper)
> and relies on multi-call epilogue for Gradient Decoupled DPO. 0.17.5
> is the last release with the simple epilogue compatible with this
> usage. Do not bump it.

---

## 2. Run the container

```bash
docker run -it --rm --gpus all --shm-size 512G --network host \
  -v $(pwd):/workspace/Teletron \
  -v /your/data/dir:/data \
  teleboost:mc0.16.1 zsh

# inside the container:
cd /workspace/Teletron
nvidia-smi  # confirm 8 GPUs visible
```

`--shm-size 512G` is required for distributed-VAE producer/consumer
sharing across DataLoader workers. `--network host` is needed if you
plan to use multi-node torchrun via TCP.

---

## 3. Smoke test on FakeDataset (no real data needed)

The fastest way to confirm the full stack works on this hardware:

```bash
# Inside the container
cd /workspace/Teletron

# Build the bench fixture (one-time; saves ~5 MB to /tmp/wan_inputs.pt)
python3 -c "
import torch
torch.manual_seed(42)
fixture = {
    'x':            torch.randn(1, 16, 2, 24, 42, dtype=torch.bfloat16),
    'y':            torch.randn(1, 20, 2, 24, 42, dtype=torch.bfloat16),
    'context':      torch.randn(1, 512, 4096,    dtype=torch.bfloat16),
    'clip_feature': torch.randn(1, 257, 1280,    dtype=torch.bfloat16),
    'timestep':     torch.randn(1,                dtype=torch.bfloat16),
}
torch.save(fixture, '/tmp/wan_inputs.pt')
print('fixture saved')
"

# Smoke test split-DPO at small scale (8-layer Wan, single iter, ~30s)
torchrun --nproc_per_node=8 --master_port=29500 \
  tests/bench_dpo_split.py \
    --mode split --num_layers 8 --fixture /tmp/wan_inputs.pt \
    --output /tmp/smoke.pt --n_iters 1
```

Expected output:

```
  iter 0: loss_c=...  loss_r=...  cur peak alloc=≈15 GB
=== mode=split num_layers=8 CP=8 ===
  peak max_memory_allocated (across ranks): ≈15 GB
```

If this works, your stack is healthy. Numerical loss values are
random-init dependent and not meaningful as a reference.

---

## 4. Real DPO training

The production entry is `examples/teleai/train_dpo.sh`. It expects two
external dependencies on `PYTHONPATH`:

```bash
# Megatron-LM at the v0.16.1 tag
git clone -b core_v0.16.1 https://github.com/NVIDIA/Megatron-LM.git /megatron
export MEGATRON_LM_DIR=/megatron

# teleai_data_tool — TeleAI internal data infrastructure (lmdb_client,
# file_client, schema.Clip etc.). Not currently open source.
# OSS users without this should subclass DPODatasetBase instead — see §5.
export TELEAI_DATA_TOOL_DIR=/path/to/teleai_data_tool

# Encoder weights (Wan 2.2 14B I2V):
# - VAE:          /path/to/Wan2.2-I2V-A14B/Wan2.1_VAE.pth
# - Text (T5):    /path/to/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth
# - Image (CLIP): /path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
# Update paths in examples/teleai/config/wan_dpo.py to match your layout.

# Launch on 8 GPUs (CP=8)
export EXPR_NAME=my_first_dpo_run
bash examples/teleai/train_dpo.sh
```

Key flags inside `train_dpo.sh` (override via env or command-line):

| Flag | Default | Notes |
|---|---|---|
| `CP` | 8 | context-parallel size; 8 covers full Wan 14B (40-head) on one node |
| `TP` | 1 | tensor-parallel; not supported for Wan currently |
| `N_VAE` | 2 | encoder rank count (out of 8) |
| `N_MOE` | 1 | DiT model copies (1 = no MoE) |
| `--bf16` | yes | required; deepspeed `communication_data_type` is wired to match |
| `--use-zero2` | yes | enables Gradient Decoupled DPO path |
| `--recompute-num-layers` | 40 | full block-recompute for memory; reduce if you have headroom |

Multi-node (DP across nodes, CP within node):

```bash
NNODES=4 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
  bash examples/teleai/train_dpo.sh
# (run on each node with the corresponding NODE_RANK)
```

---

## 5. Write your own dataset (no `teleai_data_tool` needed)

`teletron.datasets.DPODatasetBase` documents the schema your
`__getitem__` must return:

```python
import torch
from teletron.datasets import DPODatasetBase, DATASETS


class MyDPODataset(DPODatasetBase):
    """Loads pre-encoded chosen/rejected latents from disk."""

    def __init__(self, manifest_csv, **kwargs):
        import pandas as pd
        self.rows = pd.read_csv(manifest_csv).to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ch = torch.load(row["chosen_pkl"])      # pre-encoded latents
        rj = torch.load(row["rejected_pkl"])
        ctx = torch.load(row["text_emb_pkl"])   # T5 embedding

        return {
            "context": ctx,
            "chosen": {
                "latents":          ch["latents"],
                "img_clip_feature": ch["clip_feature"],
                "img_emb_y":        ch["first_frame_latent"],
            },
            "rejected": {
                "latents":          rj["latents"],
                "img_clip_feature": rj["clip_feature"],
                "img_emb_y":        rj["first_frame_latent"],
            },
        }


# Register so build_dataset("MyDPODataset") works
DATASETS.register_module(MyDPODataset)
```

Then in your config (see `examples/teleai/config/wan_dpo.py`):

```python
config = dict(
    dataset=dict(
        type="MyDPODataset",
        manifest_csv="/data/my_dpo_pairs.csv",
    ),
    # ... rest of config (model_config, encoder, etc.)
)
```

**Schema notes**:

* Tensors should be CPU + bf16 or fp32; teletron auto-casts to model
  dtype.
* Batch dim (B) is added by the DataLoader; do not prepend it.
* `chosen` and `rejected` MAY have different `T/H/W` — each branch
  goes through its own `_run_branch` forward pass. Mismatched-shape
  support is regression-tested (T-scale up to 8×, H/W up to 2×).
* For testing without real data, just use `FakeDataset` (already
  registered):

  ```python
  config = dict(dataset=dict(type="FakeDataset"), ...)
  ```

---

## 6. Common issues

**`ImportError: cannot import name 'backward_prologue'`** — your
deepspeed got bumped past 0.17.5. `pip install deepspeed==0.17.5`
inside the container.

**`KeyError: torch.bfloat16` in `ipg_buckets`** — `lr_scheduler.py`
must pass `communication_data_type=torch.bfloat16` when `args.bf16`. The
shipped code does this; only triggers if you write a custom optimizer
setup. See `teletron/train/lr_scheduler.py` for the canonical pattern.

**`ModuleNotFoundError: teleai_data_tool`** — expected for OSS users.
Verify with:

```bash
python3 -c "from teletron.datasets import FakeDataset; print(FakeDataset)"
```

This should succeed even without `teleai_data_tool` (a log line
"unavailable (No module named 'teleai_data_tool'); this is expected on
OSS installs..." is printed for each production-only dataset class).

**OOM on Wan 14B 40-layer with `--use-distributed-optimizer` (megatron)
instead of `--use-zero2`** — the megatron distributed optimizer doesn't
implement Gradient Decoupled DPO; use `--use-zero2`.

---

## 7. Verifying correctness (optional)

Element-wise verify Gradient Decoupled DPO math equivalence vs single
backward of `(loss_chosen - loss_rejected)`:

```bash
# Run both modes, dump per-rank averaged_gradients
torchrun --nproc_per_node=8 --master_port=29501 tests/bench_dpo_split.py \
    --mode split   --num_layers 8 --fixture /tmp/wan_inputs.pt \
    --output /tmp/s.pt --n_iters 1 --dump_grads /tmp/g_split

torchrun --nproc_per_node=8 --master_port=29502 tests/bench_dpo_split.py \
    --mode nosplit --num_layers 8 --fixture /tmp/wan_inputs.pt \
    --output /tmp/n.pt --n_iters 1 --dump_grads /tmp/g_nosplit

# Diff (should print PASS, max|d| ≤ 2e-4)
python3 tests/diff_split_nosplit_grads.py /tmp/g_split /tmp/g_nosplit
```

Expected:

```
TOTAL: bit-identical 60-65%   max|d|=≤3e-4   max_rel=...
PASS: max|d| < 1e-3
→ split + in-loop epilogue is math-equivalent to single backward(c+r)
  within bf16 noise.
```
