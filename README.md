# TeleBoost — GRPO for video diffusion

[![TeleBoost arXiv](https://img.shields.io/badge/TeleBoost-arXiv%202602.07595-b31b1b.svg)](https://arxiv.org/abs/2602.07595)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![BGPO arXiv](https://img.shields.io/badge/BGPO-arXiv%202511.18919-b31b1b.svg)](https://arxiv.org/abs/2511.18919)
[![VIPO arXiv](https://img.shields.io/badge/VIPO-arXiv%202511.18719-b31b1b.svg)](https://arxiv.org/abs/2511.18719)

**Paper**: [*TeleBoost: A Systematic Alignment Framework for High-Fidelity,
Controllable, and Robust Video Generation*](https://arxiv.org/abs/2602.07595)
(arXiv 2602.07595).

A production RL training stack for Wan2.1 / Wan2.2 text-to-video diffusion,
built as a recipe on top of
[`volcengine/verl`](https://github.com/volcengine/verl). It ships with five
paper-pinned algorithm variants (DanceGRPO, Flow-GRPO, GRPO-Guard, BGPO,
VIPO), a CP gradient regression test that passes bit-exact at
sp ∈ {1, 2, 4, 8}, and a four-model joint reward (aesthetic + RAFT +
VideoCLIP + VideoPhy) — all out of the box.

verl is consumed as a plain pip dependency, not vendored. Wan-specific
behavior that doesn't belong in upstream verl (model loader, FSDP wrap,
Ulysses SP patches, checkpoint compatibility) lives under
[`teleboost/`](teleboost/) and is applied at import time.

## Algorithms

DanceGRPO is the default; the others are selectable through
`TELEBOOST_METHOD` and the `ENABLE_*` flags (see [Train](#train)).

| Algorithm | Paper | What it does |
|---|---|---|
| **DanceGRPO** (default) | [arXiv 2505.07818](https://arxiv.org/abs/2505.07818) | GRPO for visual generation: per-prompt z-score advantage + σ_t = η constant SDE recast |
| **Flow-GRPO** | [arXiv 2505.05470](https://arxiv.org/abs/2505.05470) | σ_t = η·√(t/(1−t)) form + sliding-window SDE |
| **GRPO-Guard** | [arXiv 2510.22319](https://arxiv.org/abs/2510.22319) | RatioNorm (Eq. 8) + grad-reweight δ (Eq. 12) |
| **BGPO** | [arXiv 2511.18919](https://arxiv.org/abs/2511.18919) | CRT reward rerange (Eq. 4) + RAS adaptive scaling (Eq. 2) |
| **VIPO** | [arXiv 2511.18719](https://arxiv.org/abs/2511.18719) | DINOv2 PCA → per-pixel allocation map → dense advantage |

**BGPO** and **VIPO** are TeleAI papers; TeleBoost ships day-0
implementations of both.

### Reward models

The table below lists every supported reward model. The four marked ✓
are combined when `TELEBOOST_METHOD=joint`; the rest are usable as the
sole reward via `REWARD_MODEL_PATH`.

| Reward model | Paper / repo | Used in `joint`? |
|---|---|---|
| HPSv2 | [arXiv 2306.09341](https://arxiv.org/abs/2306.09341) | — |
| LAION Aesthetic predictor | [repo](https://github.com/LAION-AI/aesthetic-predictor) | ✓ |
| RAFT (optical flow) | [arXiv 2003.12039](https://arxiv.org/abs/2003.12039) | ✓ |
| VideoCLIP-XL | [arXiv 2410.00741](https://arxiv.org/abs/2410.00741) | ✓ |
| VideoPhy | [arXiv 2406.03520](https://arxiv.org/abs/2406.03520) | ✓ |
| Qwen2.5-VL-7B / 32B | (vendored vLLM rollout) | — |
| DINOv2 (advantage shaper, not a reward) | [arXiv 2304.07193](https://arxiv.org/abs/2304.07193) | — (used by VIPO) |

## Supported

| Dimension | Supported |
|---|---|
| Actor | Wan2.2-T2V-A14B (`wan_version=wan22`), Wan2.1-T2V-1.3B (`wan_version=wan21`) |
| Reward | HPSv2, Qwen2.5-VL-7B, 4-reward joint (aesthetic + RAFT + VideoCLIP + VideoPhy) |
| Algorithm | DanceGRPO (default), Flow-GRPO, GRPO-Guard, BGPO, VIPO |
| Rollout | Diffusion (actor), vLLM (Qwen reward) |
| Sequence parallel | sp ∈ {1, 2, 4, 8}; CP grad bit-exact at fp32 |
| Hardware | H800 / H100 80 GB |

## Install

Three paths, pick whichever matches your starting state:

**1. Build from our Dockerfile.** Self-contained image: NGC PyTorch 24.08
base + torch 2.6 + vllm 0.8.4 + flash-attn 2.7.4.post1 + flashinfer +
verl v0.4.0 (`--no-deps`) + Wan and reward dependencies + the hpsv2 BPE
vocab fix and a tkinter shim (hpsv2 imports tkinter at module load) +
TeleBoost as an editable install.

```bash
docker build -f docker/Dockerfile.teleboost -t teleboost:latest .
```

For region-mirrored builds (Tsinghua), pass
`--build-arg APT_SOURCE=...  --build-arg PIP_INDEX=...`; see the
Dockerfile header for the exact flags.

**2. Existing verl-compatible image.** If you already have
`verlai/verl:vllm017.latest` or similar, see [`INSTALL.md`](INSTALL.md)
— three pip commands plus an import check.

**3. Bare GPU host.** See
[`docs/install_from_scratch.md`](docs/install_from_scratch.md) — full
recipe from scratch, including the flash-attn wheel, hpsv2 packaging
fixes, and known gotchas.

## Model checkpoints

Download the actor backbone and at least one reward model before
training. Set the listed env var to the local path of the file or
directory.

| Env var | Source | Expected path target |
|---|---|---|
| `WAN_MODEL_PATH` | [`Wan-AI/Wan2.1-T2V-1.3B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) or [`Wan-AI/Wan2.2-T2V-A14B`](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | repo root directory (contains `Wan2.1_VAE.pth`, umT5 files, transformer weights) |
| `WAN_VAE_PATH` | same repo as above | `<WAN_MODEL_PATH>/Wan2.1_VAE.pth` |
| `REWARD_MODEL_PATH` | [`xswu/HPSv2`](https://huggingface.co/xswu/HPSv2) | `HPS_v2.1_compressed.pt` file |
| `PIXEL_WEIGHT_MODEL_PATH` (VIPO only) | [`facebook/dinov2-large`](https://huggingface.co/facebook/dinov2-large) | repo root (HF cache name also works) |
| `JOINT_AESTHETIC_CLIP_PATH` | [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | repo root |
| `JOINT_AESTHETIC_MODEL_PATH` | [LAION aesthetic predictor](https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth) | `sa_0_4_vit_l_14_linear.pth` file |
| `JOINT_RAFT_MODEL_PATH` | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) (`raft-things.pth`) | `raft-things.pth` file |
| `JOINT_VIDEOCLIP_MODEL_PATH` | [`alibaba-pai/VideoCLIP-XL`](https://huggingface.co/alibaba-pai/VideoCLIP-XL) | repo root |
| `JOINT_VIDEOPHY_MODEL_PATH` | [`videophysics/videocon_physics`](https://huggingface.co/videophysics/videocon_physics) | repo root |

The same `WAN_MODEL_PATH` works for 1.3B and 14B prep — the prep script
only reads the umT5 files.

## Prepare data

Every training row must carry three fields: `context_path` (positive
prompt umT5 embedding), `context_null_path` (a single shared
negative-prompt umT5 embedding for CFG), and `caption` (the original
prompt text, kept for logging and reward models that consume the raw
string). The dataset
loader fails fast if `context_null_path` is missing — without it CFG
collapses to `(1+scale) * cond` and reward variance vanishes
(advantage=0, grad_norm=0).

A row in `processed_wan_prompt.json` looks like:

```json
{
  "caption": "a panda eating bamboo, cinematic lighting",
  "context_path": "data/processed/context_0.npy",
  "context_null_path": "data/processed/context_null.npy"
}
```

One unified, idempotent prep script handles both flat prompt lists and
existing JSON layouts:

```bash
# from a plain prompts.txt (one prompt per line)
python data_preprocess/prepare_wan_data.py \
  --input prompts/mini_test.txt \
  --output_dir data/processed/ \
  --wan_model_path /path/to/Wan2.1-T2V-1.3B

# patching an existing JSON missing context_null_path
python data_preprocess/prepare_wan_data.py \
  --input data/processed/processed_wan_prompt.json \
  --output_dir data/processed/ \
  --wan_model_path /path/to/Wan2.1-T2V-1.3B
```

The script:

* accepts `.txt` (one prompt per line) or `.json` (`[{"caption": ...}, ...]`);
* only loads the umT5 encoder if something actually needs encoding;
* per row, skips T5 if `context_path` already points at an existing `.npy`;
* produces `processed_wan_prompt.json` with `caption`, `context_path`,
  and `context_null_path` all populated, plus per-row `context_<i>.npy`
  and a single shared `context_null.npy`;
* the negative prompt defaults to Wan's official Chinese template; override
  with `--negative_prompt "..."` if you need a different one.

## Train

The unified launcher `run_teleboost.sh` covers every algorithm variant
via env vars. Defaults match the upstream DanceGRPO recipe (Wan2.2-A14B,
8 GPUs, 480×832×49, 1000 steps, sampling=16); the minimal command is
just paths:

```bash
TRAIN_FILE=/path/to/processed_wan_prompt.json \
TEST_FILE=/path/to/processed_wan_prompt.json \
WAN_MODEL_PATH=/path/to/Wan2.2-T2V-A14B \
WAN_VAE_PATH=/path/to/Wan2.2-T2V-A14B/Wan2.1_VAE.pth \
REWARD_MODEL_PATH=/path/to/HPS_v2.1_compressed.pt \
bash recipe/teleboost/run_teleboost.sh
```

### Algorithm selector

| `TELEBOOST_METHOD` | Behavior | Required env vars (in addition to the core ones) |
|---|---|---|
| `default` | DanceGRPO, nothing else | — |
| `bgpo` | DanceGRPO + Bayesian-Prior reranging + RAS adaptive scaling | training rows must carry a `prior` field (see schema below) |
| `vipo` | DanceGRPO + DINOv2 dense pixel-weight broadcast | `PIXEL_WEIGHT_MODEL_PATH=...` (default `facebook/dinov2-large`) |
| `joint` | 4-reward joint (aesthetic + raft + videoclip + videophy) | `JOINT_AESTHETIC_CLIP_PATH`, `JOINT_AESTHETIC_MODEL_PATH`, `JOINT_RAFT_MODEL_PATH`, `JOINT_VIDEOCLIP_MODEL_PATH`, `JOINT_VIDEOPHY_MODEL_PATH` |

#### `prior` field schema (BGPO only)

A scalar `float` in `[0, 1]` representing the per-prompt Bayesian prior
expected reward. Used by CRT (Eq. 4) to rerange the realized reward
relative to the prior. A reasonable default is to compute it once
offline as the mean reward of K base-model rollouts on that prompt and
clip to `[0.05, 0.95]`. Example row:

```json
{
  "caption": "a panda eating bamboo",
  "context_path": "data/processed/context_0.npy",
  "context_null_path": "data/processed/context_null.npy",
  "prior": 0.42
}
```

### Common knobs

| Env var | Default | Notes |
|---|---|---|
| `N_GPUS_PER_NODE` | `8` | per-node world size |
| `SAMPLING_STEPS` | `16` | denoising steps in rollout |
| `TOTAL_TRAINING_STEPS` | `1000` | total optimizer steps |
| `VIDEO_HEIGHT`, `VIDEO_WIDTH`, `NUM_FRAMES` | `480`, `832`, `49` | resolution & frame count |
| `TRAIN_PROMPT_BSZ`, `N_RESP_PER_PROMPT` | `8`, `3` | batch shape (effective batch = bsz × n_resp) |
| `WAN_VERSION` | `wan22` | `wan21` for Wan2.1-1.3B |
| `VAL_BEFORE_TRAIN` | `False` | run validation before step 0 |
| `TELEBOOST_OUTPUT_DIR` | `./outputs` | parent for `checkpoints/` and `tensorboard/` |

### Orthogonal flags

These layer on top of `TELEBOOST_METHOD` and combine freely:

| Env var | Effect |
|---|---|
| `SP_SIZE=2` (or 4, 8) | Wan Ulysses sequence parallel; needs world_size divisible by `SP_SIZE` |
| `INIT_SAME_NOISE=False` | per-prompt responses get different starting noise — required for non-zero reward variance |
| `ENABLE_GRPOGUARD=True` | GRPO-Guard ratio_norm + grad_reweight |
| `ENABLE_FLOWGRPO=True` | Flow-GRPO SDE solver path; auto-bumps `SAMPLING_STEPS` to 4 if it was 1 |
| `ADV_ESTIMATOR=remax` | switch advantage estimator from default `grpo` to upstream verl's `remax` |

Example — sp=8 BGPO:

```bash
TELEBOOST_METHOD=bgpo SP_SIZE=8 \
TRAIN_FILE=... TEST_FILE=... WAN_MODEL_PATH=... REWARD_MODEL_PATH=... \
bash recipe/teleboost/run_teleboost.sh
```

### Multi-node

Run the same command on every node; the launcher self-routes based on
`NODE_RANK`. The master (rank 0) starts a Ray head and proceeds to the
training loop; workers join the cluster and block on `ray start`.

| Env var | Required | Notes |
|---|---|---|
| `NNODES` | yes (>1) | total node count |
| `NODE_RANK` | yes | this node's rank; `0` = master |
| `MASTER_ADDR` | yes | hostname / IP of the master, reachable from every worker |
| `MASTER_PORT` | no | Ray head port; default `6379` |

Example — 4 nodes × 8 GPUs (32 GPUs total):

```bash
# on the master (NODE_RANK=0):
NNODES=4 NODE_RANK=0 MASTER_ADDR=node-0.example.com \
TRAIN_FILE=... TEST_FILE=... WAN_MODEL_PATH=... \
WAN_VAE_PATH=... REWARD_MODEL_PATH=... \
bash recipe/teleboost/run_teleboost.sh

# on each worker (NODE_RANK=1, 2, 3):
NNODES=4 NODE_RANK=1 MASTER_ADDR=node-0.example.com \
TRAIN_FILE=... TEST_FILE=... WAN_MODEL_PATH=... \
WAN_VAE_PATH=... REWARD_MODEL_PATH=... \
bash recipe/teleboost/run_teleboost.sh
```

`NNODES=1` (the default) skips the Ray head/worker step entirely;
`main_teleboost` calls `ray.init()` itself for single-node runs.

## Multi-reward joint setup

When `TELEBOOST_METHOD=joint`, four reward models (aesthetic, RAFT,
VideoCLIP, VideoPhy) co-exist on every actor GPU and compute concurrently.

![Co-located reward + MPS-parallel multi-reward](docs/figures/colocate_mps.png)

Two architectural choices matter here:

**1. Co-located reward + actor.** Reward workers share the actor GPUs
(`dp_fraction=1.0, rank_offset=0` for every model). The earlier
disaggregated layout (separate reward GPUs) was removed because
`_allgather_rewards` calls `dist.all_gather` on the default world process
group — any `dp_size != world_size` triggers a length-mismatch crash.
Co-location also eliminates the rollout-idle / reward-idle stalls in the
left half of the figure above. See
[`recipe/teleboost/config/teleboost_trainer.yaml`](recipe/teleboost/config/teleboost_trainer.yaml#L172)
for the dispatch comment.

**2. MPS-parallel multi-reward.** Four reward forwards run serially on
one GPU would block the actor on a wall-clock sum of all four model
forwards. NVIDIA CUDA MPS lets each reward model occupy a fixed thread
percentage of the same GPU; the four models compute concurrently and the
joint forward finishes in roughly the time of the *slowest* single model,
not the sum. Default thread allocation:

| Model | Approx. params | `mps_percentage` |
|---|---|---|
| `aesthetic` (LAION ViT-L/14 + linear head) | 5.3M | 20 |
| `raft` (RAFT optical flow) | 5.3M | 30 |
| `videoclip` (VideoCLIP-XL) | 0.42B | 25 |
| `videophy` (videocon_physics) | 7B | 25 |

The four percentages should sum to ≤100; the loader does not enforce
this, but CUDA MPS itself caps the total at 100% effective. Sums <100
leave the remainder unallocated (the kernel launcher uses whatever
threads are free, with no guarantee).

Per-model Hydra overrides:

```bash
# tune one model's share (rebalance the others so the sum stays ≤100)
bash recipe/teleboost/run_teleboost.sh \
  reward_model.joint.mps.model_percentages.videophy=40 \
  reward_model.joint.mps.model_percentages.videoclip=15

# disable MPS — falls back to within-GPU serial
bash recipe/teleboost/run_teleboost.sh \
  reward_model.joint.mps.enabled=false
```

Aggregation across the four rewards is configurable
(`reward_model.joint.aggregation`, default `weighted_sum`); per-model
weights live under `reward_model.joint.models.<name>.weight`.

## Repository layout

```
recipe/teleboost/                     Recipe (entry point + workers + scripts)
├── main_teleboost.py                     Hydra entry; spawns Ray workers
├── teleboost_ray_trainer.py              RayPPOTrainer subclass
├── dp_actor.py                           DataParallelPPOActor subclass
├── teleboost_fsdp_worker.py              7 reward workers + DiffusionActorRolloutRefWorker subclass
├── unified_reward_worker.py              plugin-style reward worker
├── reward_models/                        reward registry + 5 plugins + composite + dynamic_joint
├── algorithms/                           paper-pinned algorithm modules
├── config/teleboost_trainer.yaml         Hydra config (grpo_guard, flow_grpo, joint.* fields)
└── run_teleboost.sh                      unified env-driven launcher

teleboost/                            TeleBoost-only extensions
├── models/transformers/wan.py            Wan Ulysses SP forward + helper patches
├── models/transformers/wan22.py          Wan2.2 dual-model wrapper
├── workers/rollout/diffusion_rollout.py  diffusion rollout (replaces vllm rollout)
├── workers/sharding_manager/diffusion.py FSDP sharding manager for diffusion rollout
├── utils/diffusion_ulysses.py            SP > 1 split/gather autograd Functions
└── patches/                              runtime monkey-patches over verl
    ├── ulysses_cp_fix.py                 CP grad reduce fix (modulation params)
    ├── wan_ulysses.py                    inject Wan SP helpers into verl.utils.ulysses
    ├── wan_save_compat.py                FrozenDict.save_pretrained no-op for save_checkpoint
    └── debug_extras.py                   marked_timer / simple_timer / ProfilerConfig backports

wan/                                  Wan2.1 / Wan2.2 backbone (vendored, lightly patched)
data_preprocess/prepare_wan_data.py   umT5 prompt-embedding prep (idempotent)
models/videoalign/                    Qwen2VL-based video-alignment reward model trainer (standalone)
prompts/                              prompt lists for preprocess
docs/                                 docs
tests/special_distributed/            distributed regression tests (CP grad reduce)
```

`teleboost/__init__.py` runs `teleboost.patches.apply()` at import time, so
any process that does `import teleboost` (or imports anything under
`recipe.teleboost.*`, which imports teleboost) ends up with the patches
applied automatically.

## Documentation

| File | Topic |
|---|---|
| [`INSTALL.md`](INSTALL.md) | Fast install path on a verl-ready Docker image |
| [`docs/install_from_scratch.md`](docs/install_from_scratch.md) | Bare-host install + every gotcha |
| [`requirements-pinned.txt`](requirements-pinned.txt) | Full pin file (every transitive dep) |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to send a PR, run tests, file an issue |

## Citation

If TeleBoost is useful for your work, please cite the framework paper:

```bibtex
@article{teleboost2026,
  title   = {TeleBoost: A Systematic Alignment Framework for High-Fidelity,
             Controllable, and Robust Video Generation},
  author  = {Liang, Yuanzhi and Wu, Xuan'er and Liu, Yirui and Fang, Yijie
             and Fan, Yizhen and Hao, Ke and Li, Rui and Liu, Ruiying
             and Ni, Ziqi and Yu, Peng and Wang, Yanbo and Huang, Haibin
             and Weng, Qizhen and Zhang, Chi and Li, Xuelong},
  journal = {arXiv preprint arXiv:2602.07595},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.07595}
}
```

The two TeleAI algorithms shipped day-0 in this codebase:

```bibtex
@article{liu2025bgpo,
  title   = {Learning What to Trust: Bayesian Prior-Guided Optimization
             for Visual Generation},
  author  = {Liu, Ruiying and Liang, Yuanzhi and Huang, Haibin
             and Yu, Tianshu and Zhang, Chi},
  journal = {arXiv preprint arXiv:2511.18919},
  year    = {2025},
  url     = {https://arxiv.org/abs/2511.18919}
}

@article{ni2025vipo,
  title   = {Seeing What Matters: Visual Preference Policy Optimization
             for Visual Generation},
  author  = {Ni, Ziqi and Liang, Yuanzhi and Li, Rui and Zhou, Yi
             and Huang, Haibin and Zhang, Chi and Li, Xuelong},
  journal = {arXiv preprint arXiv:2511.18719},
  year    = {2025},
  url     = {https://arxiv.org/abs/2511.18719}
}
```

For the upstream papers TeleBoost builds on (DanceGRPO, Flow-GRPO,
GRPO-Guard, GRPO), see [`CITATION.cff`](CITATION.cff) — ready for
`cffconvert` / GitHub's "Cite this repository" widget.

## License & acknowledgments

Apache 2.0 (see [`LICENSE`](LICENSE) and [`Notice.txt`](Notice.txt)).

This codebase stands on the shoulders of several open-source projects.
We thank the authors of all of the following.

**Framework**

* [**volcengine/verl**](https://github.com/volcengine/verl) (Apache 2.0,
  Bytedance Seed) — the PPO / GRPO training engine. We pin
  `verl@v0.4.0` and consume it as a pip dependency rather than vendoring;
  recipe-level extensions live under
  [`recipe/teleboost/`](recipe/teleboost/) and
  [`teleboost/`](teleboost/).

**Algorithms**

The algorithm modules under
[`recipe/teleboost/algorithms/`](recipe/teleboost/algorithms/) are
paper-faithful translations of:

* GRPO ([DeepSeekMath, arXiv 2402.03300](https://arxiv.org/abs/2402.03300)) — referenced for the base objective; not a standalone training mode here
* DanceGRPO ([arXiv 2505.07818](https://arxiv.org/abs/2505.07818))
* Flow-GRPO ([arXiv 2505.05470](https://arxiv.org/abs/2505.05470))
* GRPO-Guard ([arXiv 2510.22319](https://arxiv.org/abs/2510.22319))
* BGPO ([arXiv 2511.18919](https://arxiv.org/abs/2511.18919))
* VIPO ([arXiv 2511.18719](https://arxiv.org/abs/2511.18719))

See each module's docstring for the equation pin.

**Models — actor backbones**

* [**Wan-Video/Wan2.1**](https://github.com/Wan-Video/Wan2.1) and
  [**Wan-Video/Wan2.2**](https://github.com/Wan-Video/Wan2.2) —
  text-to-video diffusion actor. The `wan/` directory bundles their
  model code; the FSDP / Ulysses-SP wrap and CP-grad-reduce fix in
  [`teleboost/patches/`](teleboost/patches/) are our additions.

**Reward models**

* [**tgxs002/HPSv2**](https://github.com/tgxs002/HPSv2)
  ([arXiv 2306.09341](https://arxiv.org/abs/2306.09341))
* [**alibaba-pai/VideoCLIP-XL**](https://huggingface.co/alibaba-pai/VideoCLIP-XL)
  ([arXiv 2410.00741](https://arxiv.org/abs/2410.00741))
* [**LAION-AI/aesthetic-predictor**](https://github.com/LAION-AI/aesthetic-predictor)
* [**princeton-vl/RAFT**](https://github.com/princeton-vl/RAFT)
  ([arXiv 2003.12039](https://arxiv.org/abs/2003.12039))
* [**videophysics/videocon_physics**](https://huggingface.co/videophysics/videocon_physics)
  ([arXiv 2406.03520](https://arxiv.org/abs/2406.03520))
* [**facebook/dinov2**](https://github.com/facebookresearch/dinov2)
  ([arXiv 2304.07193](https://arxiv.org/abs/2304.07193)) — used by VIPO
  for per-pixel allocation maps.

**Compute & systems**

* [**vllm-project/vllm**](https://github.com/vllm-project/vllm) — Qwen
  reward worker rollout.
* [**Dao-AILab/flash-attention**](https://github.com/Dao-AILab/flash-attention)
  — flash-attn 2.7.4.post1 used in actor + reward rollouts.
* [**flashinfer-ai/flashinfer**](https://github.com/flashinfer-ai/flashinfer)
  — vLLM kernel backend.
