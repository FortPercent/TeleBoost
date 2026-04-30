# TeleBoost — Memory-Efficient Video DPO Training

Training framework for video diffusion models, featuring **Gradient
Decoupled DPO** — a per-branch backward + immediate reduce-scatter
pattern that **unlocks Wan 14B 40-layer DPO on 8×H800 80GB**, a config
that crashes (OOM) on standard DPO implementations.

Built on [megatron-core 0.16.1](https://github.com/NVIDIA/Megatron-LM/tree/core_v0.16.1)
+ DeepSpeed ZeRO-2 + bf16 + recompute + context-parallelism. Production
in TeleAI internally for Wan-family training; this is the OSS release.

---

## Headline numbers — Wan 14B DPO on 8×H800 80GB

| `num_layers` | Standard DPO | **Gradient Decoupled DPO** | Δ |
|---|---|---|---|
| 4 (toy)  | 9.15 GB | 7.72 GB | **−15.6%** |
| 36 (max-fit) | 65.71 GB | 58.71 GB | **−10.6%** |
| **40 (production)** | **OOM (>80 GB)** | **65.17 GB** | **qualitative** ✅ |

**Math equivalence verified element-wise** at Wan 14B 36-layer scale:
14.78 billion gradient elements compared, max\|Δgrad\| = 2.4e-4, well
within bf16 ULP. The split pattern is mathematically identical to
standard `(loss_chosen − loss_rejected).backward()` (chain rule +
reduce_scatter linearity); only peak memory changes.

How it works:

```python
# Standard DPO: both branches' grads alive together at peak
(coeff * loss_chosen - coeff * loss_rejected).backward()
optimizer.epilogue()

# Gradient Decoupled DPO: per-branch backward, immediate reduce-scatter
for t in [-coeff * loss_rejected, coeff * loss_chosen]:
    optimizer.backward(t)
    optimizer.overlapping_partition_gradients_reduce_epilogue()
    # → my-shard 1/N of t's gradient written to averaged_gradients
    # → full-shape grad tensor freed before next backward starts
```

---

## Quickstart

See [QUICKSTART.md](QUICKSTART.md) for the full walkthrough.

```bash
# 1. Build the image (Hopper / SM 9.0; ~80 min on a clean cache including
#    flash-attn 2 + flash-attn 3 source build)
docker build -t teleboost:mc0.16.1 .

# 2. Run on 8 H100 / H200 / H800
docker run -it --gpus all --shm-size 512G \
    -v $(pwd):/workspace/Teletron \
    -v /path/to/your/data:/data \
    teleboost:mc0.16.1

# 3. Smoke test inside the container
cd /workspace/Teletron
torchrun --nproc_per_node=8 examples/wan/pretrain_wan2_2.py \
    --dataset-type FakeDataset --bf16 --use-zero2 ...
# (full args in QUICKSTART.md)

# 4. Real DPO training
export MEGATRON_LM_DIR=/path/to/Megatron-LM
export TELEAI_DATA_TOOL_DIR=/path/to/teleai_data_tool   # for production data
bash examples/teleai/train_dpo.sh
```

For users without `teleai_data_tool` (the internal data-infrastructure
package), subclass `teletron.datasets.DPODatasetBase` and register your
own dataset — see QUICKSTART for the 30-line template.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  examples/teleai/train_dpo.sh                               │
│  └─→ examples/teleai/pretrain_dpo_i2v.py                    │
│      └─→ teletron.train.Trainer                             │
│           ├─ ParallelWanModel (40-layer DiT)                │
│           ├─ DistributedVAE (text + image + video encoder)  │
│           └─ DeepSpeedZeroOptimizer (ZeRO-2 partition_grads)│
│                └─ deepspeed_backward_step (split DPO path)  │
└─────────────────────────────────────────────────────────────┘
       │
       ├─ CP=8  : context parallelism via Ulysses (head-dim sharding)
       ├─ ZeRO-2: optimizer state partitioned across DP_with_CP group
       ├─ recompute=full+block : every block input checkpointed
       └─ bf16 : mixed precision with fp32 master weights
```

---

## Parallelism configuration

| Flag | Description |
|---|---|
| `--context-parallel-size` (`CP`) | sequence parallelism within node; head-dim sharded (Ulysses) |
| `--tensor-model-parallel-size` (`TP`) | tensor-parallel; weight-dim sharded |
| `--use-zero2` | enable DeepSpeedZeroOptimizer + Gradient Decoupled DPO path |
| `--distributed-vae` | run encoder on dedicated ranks, freeing DiT ranks |
| `--distributed-vae-world-size` (`N_VAE`) | encoder rank count |
| `--consumer-models-num` (`N_MOE`) | DiT model copies (1 = no MoE) |

Constraint: `(TP × CP)` must divide `num_attention_heads`. For Wan 14B
(40 heads), valid CP×TP combos are 1, 2, 4, 5, 8, 10, 20, 40.

---

## Supported models

| Model | Params | dim | heads | layers |
|---|---|---|---|---|
| Wan2.1 / Wan2.2 (T2V/I2V) | 14B | 5120 | 40 | 40 |
| Wan2.1 1.3B | 1.3B | 1536 | 12 | 30 |
| CausalWan2.1 1.3B | 1.3B | 1536 | 12 | 30 |

Production focus is **Wan 14B I2V DPO** (`examples/teleai/`); other
variants live under `examples/wan/`.

---

## Common features

- **EMA** (`--with-ema --ema-decay 0.9999`): EMA weights sharded across
  DP for low memory overhead.
- **Checkpoint resume** (`--save / --load --save-interval`): full
  optim-state + RNG-state included; `--data-parallel-random-init`
  recommended for stable DPO training (per-DP-rank timestep RNG).
- **`torch.compile` for VAE** (`torch_compile=True` in encoder config):
  20-40% encoder speed-up.
- **flash-attn 2** auto-used; **flash-attn 3** on Hopper auto-detected
  via `transformer_engine`.

---

## Hard requirements

- **GPU**: SM 9.0 (H100 / H200 / H800) recommended; SM 8.0+ works with
  `--build-arg BUILD_FA3=0`.
- **CUDA**: 13.0 (NGC 25.09); driver compatible with cu13 stack.
- **Python**: 3.12.

The [Dockerfile](Dockerfile) bakes everything ABI-aligned. Do not
upgrade torch / transformer_engine / apex / deepspeed inside the
image — see comments at the top of `Dockerfile` and `requirements.txt`
for the rationale (deepspeed 0.17.6+ in particular breaks the
multi-call epilogue that Gradient Decoupled DPO depends on).

---

## Repository layout

```
Teletron/
├── README.md           ← you are here
├── QUICKSTART.md       ← full setup + first-run guide
├── Dockerfile          ← reproducible NGC 25.09 + flash-attn 2/3
├── requirements.txt    ← pinned Python deps; deepspeed==0.17.5
├── examples/
│   ├── teleai/         ← Wan 14B DPO production entry
│   │   ├── train_dpo.sh
│   │   ├── pretrain_dpo_i2v.py
│   │   └── config/wan_dpo.py
│   └── wan/            ← Wan T2V/I2V non-DPO entries
│       ├── pretrain_wan.py
│       ├── pretrain_wan2_2.py
│       └── ...
├── teletron/
│   ├── train/
│   │   ├── utils.py            ← deepspeed_backward_step (split path)
│   │   ├── lr_scheduler.py     ← optimizer wiring
│   │   └── trainer.py
│   ├── models/wan/             ← ParallelWanModel
│   ├── core/context_parallel/  ← CP all-to-all
│   └── datasets/
│       ├── dpo_base.py         ← DPODatasetBase  ← OSS users subclass
│       ├── fake_dataset.py     ← FakeDataset for smoke tests
│       └── build.py            ← lazy registry
└── tests/
```

---

## Citation

```bibtex
@misc{teleboost2026,
  title  = {TeleBoost: Gradient Decoupled DPO for memory-efficient
            video diffusion model training},
  author = {TeleAI Infra Team},
  year   = {2026},
  url    = {https://github.com/FortPercent/TeleBoost},
}
```
