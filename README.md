<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FortPercent/TeleBoost/release-staging/documents/figures/logo_teleboost.jpeg">
    <img alt="TeleBoost" src="https://raw.githubusercontent.com/FortPercent/TeleBoost/release-staging/documents/figures/logo_teleboost.jpeg" width="55%">
  </picture>
</p>
<h3 align="center">
GRPO post-training for video diffusion models.
</h3>

<p align="center">
  <a href="https://tele-ai.github.io/TeleBoost/"><img alt="Project page" src="https://img.shields.io/badge/Project_page-tele--ai.github.io-4C1?labelColor=555555"></a>
  <a href="https://arxiv.org/abs/2602.07595"><img alt="TeleBoost arXiv" src="https://img.shields.io/badge/TeleBoost-arXiv%202602.07595-B31B1B?labelColor=555555"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-2196F3?labelColor=555555"></a>
  <a href="https://arxiv.org/abs/2511.18919"><img alt="BGPO arXiv" src="https://img.shields.io/badge/BGPO-arXiv%202511.18919-B31B1B?labelColor=555555"></a>
  <a href="https://arxiv.org/abs/2511.18719"><img alt="VIPO arXiv" src="https://img.shields.io/badge/VIPO-arXiv%202511.18719-B31B1B?labelColor=555555"></a>
</p>

TeleBoost (GRPO branch) is a **production RL training stack for Wan2.1 /
Wan2.2 text-to-video diffusion**, built as a recipe layer on top of
[`volcengine/verl`](https://github.com/volcengine/verl).

* 🎛️ **Five algorithms** — DanceGRPO, Flow-GRPO, GRPO-Guard, BGPO, VIPO
* 🎬 **Drop-in sequence parallel** — Wan Ulysses CP for long-video training
* 🧩 **MPS-parallel multi-reward** — N rewards concurrent on one GPU; wall-time ≈ max(model), not sum
* 🆕 **Day-0 BGPO + VIPO** — TeleAI papers, implemented on release

Inspired by the GRPO recipe in [`volcengine/verl`](https://github.com/volcengine/verl),
given a dedicated home to evolve around video-diffusion-specific constraints.
Used internally at TeleAI for Wan-family alignment.

<p align="center">
  <img src="docs/figures/colocate_mps.png" alt="Two independent throughput optimizations. Left: co-located reward — reward workers share the actor GPUs, eliminating the dedicated reward-rank's rollout-idle and training-idle gaps. Right: MPS-parallel multi-reward — N reward models compute concurrently on the same GPU via CUDA MPS, with wall-time bounded by the slowest model rather than the sum." width="820"/>
</p>

<p align="center"><sub><i>
Two independent throughput optimizations. <b>Left</b>: co-located reward
shares the actor GPUs, eliminating the dedicated reward-rank's
rollout-idle and training-idle gaps. <b>Right</b>: CUDA MPS — N reward
models compute concurrently on the same GPU, wall-time ≈ max(model)
instead of sum.
</i></sub></p>

---

## What's new from TeleAI

Three TeleAI contributions ship in this repo: two day-0 algorithm
papers (VIPO, BGPO) and a systems-level throughput optimization
(co-located reward + MPS). Motivation in one paragraph each; headline
numbers and recipes are in [Headline matrix](#headline-matrix).

### VIPO — Visual In-Pixel Policy Optimization ([arXiv 2511.18719](https://arxiv.org/abs/2511.18719))

GRPO for visual generation traditionally drives the policy with a single
scalar reward, which makes credit assignment hard at the pixel / region
level and tends to optimize global signals at the expense of local
fidelity. VIPO introduces a **Pixel Score Map (PSM)** module that
converts the scalar advantage into a pixel- or region-level advantage
map, giving the policy structured spatial feedback. On both image- and
video-generation benchmarks VIPO delivers significant gains over
scalar-advantage GRPO, pointing to structured spatial feedback as a new
direction for visual-generation post-training.

### BGPO — Bayesian-Prior Group Optimization ([arXiv 2511.18919](https://arxiv.org/abs/2511.18919))

Reward models for image- and video-generation RL post-training are noisy
and prone to alignment bias, which can mislead GRPO updates. BGPO models
that reward uncertainty as a **Bayesian prior**, assigns per-sample
trust weights (RAS, Eq. 2), and recalibrates the reward signal (CRT
rerange, Eq. 4). The result is more stable training, tighter alignment
to intended preferences, and faster convergence of the generative
policy.

### Co-located reward + MPS-parallel multi-reward (systems)

Multi-reward GRPO post-training has two practical bottlenecks: a
dedicated reward-rank GPU group sits idle during rollout / training
swaps, and N sequential reward forwards make joint wall-time scale with
the **sum** rather than the **max** of model latencies. TeleBoost ships
two independent fixes — **co-located reward** (reward workers share the
actor GPUs, eliminating the idle reward rank) and **MPS-parallel
multi-reward** (N reward models execute concurrently on the same GPU
via CUDA MPS, wall-time ≈ max(model) instead of sum). Both are on by
default in joint mode; see [Multi-reward joint](#multi-reward-joint-teleboost_methodjoint)
for the recipe and the figure at the top of this README for the
illustration.

---

## Headline matrix

| | Variant | Paper | What it does |
|---|---|---|---|
| 🟢 default | **DanceGRPO** | [arXiv 2505.07818](https://arxiv.org/abs/2505.07818) | GRPO for visual generation: per-prompt z-score advantage + σ_t = η constant SDE recast |
| | Flow-GRPO | [arXiv 2505.05470](https://arxiv.org/abs/2505.05470) | σ_t = η·√(t/(1−t)) form + sliding-window SDE |
| | GRPO-Guard | [arXiv 2510.22319](https://arxiv.org/abs/2510.22319) | RatioNorm (Eq. 8) + grad-reweight δ (Eq. 12) |
| 🔵 TeleAI | **BGPO** | [arXiv 2511.18919](https://arxiv.org/abs/2511.18919) | CRT reward rerange (Eq. 4) + RAS adaptive scaling (Eq. 2) |
| 🔵 TeleAI | **VIPO** | [arXiv 2511.18719](https://arxiv.org/abs/2511.18719) | DINOv2 PCA → per-pixel allocation map → dense advantage |

### Reward models

| Reward model | Paper / repo |
|---|---|
| HPSv2 | [arXiv 2306.09341](https://arxiv.org/abs/2306.09341) |
| LAION Aesthetic predictor | [repo](https://github.com/LAION-AI/aesthetic-predictor) |
| RAFT (optical flow) | [arXiv 2003.12039](https://arxiv.org/abs/2003.12039) |
| VideoCLIP-XL | [arXiv 2410.00741](https://arxiv.org/abs/2410.00741) |
| VideoPhy | [arXiv 2406.03520](https://arxiv.org/abs/2406.03520) |
| Qwen2.5-VL-7B / 32B | (vendored vLLM rollout) |
| DINOv2 (advantage shaper for VIPO) | [arXiv 2304.07193](https://arxiv.org/abs/2304.07193) |

### Supported configurations

| Dimension | Supported |
|---|---|
| Actor | Wan2.2-T2V-A14B, Wan2.1-T2V-1.3B |
| Reward | HPSv2, Qwen2.5-VL-7B, joint reward (4 default models) |
| Algorithm | DanceGRPO (default), Flow-GRPO, GRPO-Guard, BGPO, VIPO |
| Rollout | Diffusion (actor), vLLM (Qwen reward) |
| Sequence parallel | Supported |
| Hardware | H800 / H100 80 GB |

---

## Quickstart

```bash
# 1. Build the image (NGC PyTorch 24.08 + torch 2.6 + vllm 0.8.4 +
#    flash-attn 2.7.4.post1 + verl@v0.4.0 + reward stack)
docker build -f docker/Dockerfile.teleboost -t teleboost:latest .

# 2. Prep data (idempotent — accepts plain prompts.txt or existing JSON)
python data_preprocess/prepare_wan_data.py \
  --input prompts/mini_test.txt \
  --output_dir data/processed/ \
  --wan_model_path /path/to/Wan2.1-T2V-1.3B

# 3. Train — DanceGRPO defaults (Wan2.2-A14B, 8 GPUs, 480×832×49, 1000 steps)
TRAIN_FILE=data/processed/processed_wan_prompt.json \
TEST_FILE=data/processed/processed_wan_prompt.json \
WAN_MODEL_PATH=/path/to/Wan2.2-T2V-A14B \
WAN_VAE_PATH=/path/to/Wan2.2-T2V-A14B/Wan2.1_VAE.pth \
REWARD_MODEL_PATH=/path/to/HPS_v2.1_compressed.pt \
bash recipe/teleboost/run_teleboost.sh
```

For verl-prebuilt-image and bare-host paths, see [`INSTALL.md`](INSTALL.md)
and [`docs/install_from_scratch.md`](docs/install_from_scratch.md).
For region-mirrored Docker builds (Tsinghua), pass
`--build-arg APT_SOURCE=...  --build-arg PIP_INDEX=...`.

---

## Data schema

Every training row carries three fields:

| Field | Meaning |
|---|---|
| `caption` | Original prompt text — kept for logging and reward models that consume the raw string |
| `context_path` | umT5 embedding of the **positive** prompt |
| `context_null_path` | umT5 embedding of the shared **negative** prompt (CFG) |

The dataset loader fails fast if `context_null_path` is missing — without
it CFG collapses to `(1+scale)·cond` and reward variance vanishes
(`advantage=0`, `grad_norm=0`).

```json
{
  "caption": "a panda eating bamboo, cinematic lighting",
  "context_path": "data/processed/context_0.npy",
  "context_null_path": "data/processed/context_null.npy"
}
```

The prep script (`data_preprocess/prepare_wan_data.py`):

* Accepts `.txt` (one prompt per line) **or** `.json` (`[{"caption": ...}, ...]`);
* Loads the umT5 encoder lazily — only if something actually needs encoding;
* Per row, skips T5 if `context_path` already points at an existing `.npy`;
* Writes `processed_wan_prompt.json` + per-row `context_<i>.npy` +
  one shared `context_null.npy`;
* Uses Wan's official Chinese negative-prompt template by default; override
  with `--negative_prompt "..."`.

---

## Model checkpoints

| Env var | Source | Expected target |
|---|---|---|
| `WAN_MODEL_PATH` | [`Wan-AI/Wan2.1-T2V-1.3B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) or [`Wan-AI/Wan2.2-T2V-A14B`](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | repo root (contains `Wan2.1_VAE.pth`, umT5 files, transformer weights) |
| `WAN_VAE_PATH` | same repo as above | `<WAN_MODEL_PATH>/Wan2.1_VAE.pth` |
| `REWARD_MODEL_PATH` | [`xswu/HPSv2`](https://huggingface.co/xswu/HPSv2) | `HPS_v2.1_compressed.pt` |
| `PIXEL_WEIGHT_MODEL_PATH` (VIPO) | [`facebook/dinov2-large`](https://huggingface.co/facebook/dinov2-large) | repo root (HF cache name also works) |
| `JOINT_AESTHETIC_CLIP_PATH` | [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | repo root |
| `JOINT_AESTHETIC_MODEL_PATH` | [LAION aesthetic predictor](https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth) | `sa_0_4_vit_l_14_linear.pth` |
| `JOINT_RAFT_MODEL_PATH` | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | `raft-things.pth` |
| `JOINT_VIDEOCLIP_MODEL_PATH` | [`alibaba-pai/VideoCLIP-XL`](https://huggingface.co/alibaba-pai/VideoCLIP-XL) | repo root |
| `JOINT_VIDEOPHY_MODEL_PATH` | [`videophysics/videocon_physics`](https://huggingface.co/videophysics/videocon_physics) | repo root |

The same `WAN_MODEL_PATH` works for both 1.3B and 14B prep — the prep
script only reads the umT5 files.

---

## Algorithm selector

| `TELEBOOST_METHOD` | Behavior | Extra env vars |
|---|---|---|
| `default` | DanceGRPO, nothing else | — |
| `bgpo` | DanceGRPO + Bayesian-Prior reranging + RAS adaptive scaling | rows must carry `prior` field (see below) |
| `vipo` | DanceGRPO + DINOv2 dense pixel-weight broadcast | `PIXEL_WEIGHT_MODEL_PATH=...` (default `facebook/dinov2-large`) |
| `joint` | 4-reward joint (aesthetic + RAFT + VideoCLIP + VideoPhy) | `JOINT_AESTHETIC_*`, `JOINT_RAFT_*`, `JOINT_VIDEOCLIP_*`, `JOINT_VIDEOPHY_*` |

#### `prior` field schema (BGPO only)

A scalar `float` ∈ [0, 1] — the per-prompt Bayesian prior expected reward.
Used by CRT (Eq. 4) to rerange the realized reward relative to the prior.
Reasonable default: compute once offline as the mean reward of K
base-model rollouts on that prompt and clip to `[0.05, 0.95]`.

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

---

## Multi-reward joint (`TELEBOOST_METHOD=joint`)

When `TELEBOOST_METHOD=joint`, four reward models co-exist on every actor
GPU and compute *concurrently*. Two architectural choices matter:

**1. Co-located reward + actor.** Reward workers share the actor GPUs
(`dp_fraction=1.0, rank_offset=0` for every model). The earlier
disaggregated layout (separate reward GPUs) was removed because
`_allgather_rewards` calls `dist.all_gather` on the default world process
group — any `dp_size != world_size` triggers a length-mismatch crash.
Co-location also eliminates the rollout-idle / reward-idle stalls in the
left half of the figure above.
See [`recipe/teleboost/config/teleboost_trainer.yaml`](recipe/teleboost/config/teleboost_trainer.yaml#L172).

**2. MPS-parallel multi-reward.** Four reward forwards run *serially* on
one GPU would block the actor on the wall-clock sum of all four model
forwards. CUDA MPS lets each reward model occupy a fixed thread
percentage of the same GPU; the four compute concurrently, and the joint
forward finishes in roughly `max(model)`, not `Σ(model)`. Per-model
thread shares and aggregation weights are Hydra-configurable under
`reward_model.joint.*` — see
[`recipe/teleboost/config/teleboost_trainer.yaml`](recipe/teleboost/config/teleboost_trainer.yaml).

---

## Hard requirements

- **GPU**: H800 / H100 80 GB; SM 8.0+ in principle, but flash-attn 2.7.4
  wheel is built for SM 8.0 / 9.0.
- **CUDA**: cu12 stack (NGC PyTorch 24.08 base); driver compatible with
  cu12.
- **Python**: 3.10 (NGC base).
- **Pinned**: `verl@v0.4.0`, `transformers<5`, `vllm==0.8.4`,
  `flash-attn==2.7.4.post1`.

The [`docker/Dockerfile.teleboost`](docker/Dockerfile.teleboost) bakes
everything ABI-aligned (incl. the hpsv2 BPE-vocab fix and a tkinter shim
that hpsv2 imports at module load). Do not casually upgrade torch /
transformer_engine / flash-attn / verl inside the image — see
[`requirements-pinned.txt`](requirements-pinned.txt) for the rationale.

---

## Repository layout

```
recipe/teleboost/                     Recipe (entry + workers + scripts)
├── main_teleboost.py                     Hydra entry; spawns Ray workers
├── teleboost_ray_trainer.py              RayPPOTrainer subclass
├── dp_actor.py                           DataParallelPPOActor subclass
├── teleboost_fsdp_worker.py              7 reward workers + DiffusionActor… subclass
├── unified_reward_worker.py              plugin-style reward worker
├── reward_models/                        registry + 5 plugins + composite + dynamic_joint
├── algorithms/                           paper-pinned algorithm modules
├── config/teleboost_trainer.yaml         Hydra config
└── run_teleboost.sh                      unified env-driven launcher

teleboost/                            TeleBoost-only extensions
├── models/transformers/wan.py            Wan Ulysses SP forward + helpers
├── models/transformers/wan22.py          Wan2.2 dual-model wrapper
├── workers/rollout/diffusion_rollout.py  diffusion rollout (replaces vllm rollout)
├── workers/sharding_manager/diffusion.py FSDP sharding manager
├── utils/diffusion_ulysses.py            SP > 1 split/gather autograd Functions
└── patches/                              runtime monkey-patches over verl
    ├── ulysses_cp_fix.py                 CP grad reduce fix (modulation params)
    ├── wan_ulysses.py                    inject Wan SP helpers into verl.utils.ulysses
    ├── wan_save_compat.py                FrozenDict.save_pretrained no-op
    └── debug_extras.py                   marked_timer / simple_timer / ProfilerConfig backports

wan/                                  Wan2.1 / Wan2.2 backbone (vendored, lightly patched)
data_preprocess/prepare_wan_data.py   umT5 prompt-embedding prep (idempotent)
models/videoalign/                    Qwen2VL video-alignment reward trainer (standalone)
prompts/                              prompt lists for preprocess
docs/                                 docs + figures
tests/special_distributed/            distributed regression tests (CP grad reduce)
```

`teleboost/__init__.py` runs `teleboost.patches.apply()` at import time,
so any process that does `import teleboost` (or imports anything under
`recipe.teleboost.*`, which imports teleboost) ends up with the patches
applied automatically.

---

## Documentation

| File | Topic |
|---|---|
| [`INSTALL.md`](INSTALL.md) | Fast install on a verl-ready Docker image |
| [`docs/install_from_scratch.md`](docs/install_from_scratch.md) | Bare-host install + every gotcha |
| [`requirements-pinned.txt`](requirements-pinned.txt) | Full pin file (every transitive dep) |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to send a PR, run tests, file an issue |

---

## Roadmap

* **Day-0 support for new TeleAI algorithms** — keep the BGPO / VIPO
  cadence: new TeleAI alignment papers land in this codebase on release
* **BAGEL** — unified multi-modal understanding + generation as an actor
* **World models** — extend the actor side beyond pure T2V video diffusion

Issues / PRs welcome.

---

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

---

## License & Acknowledgments

This project is licensed under the Apache License 2.0 — see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

This codebase builds on the following open-source projects. We thank
their authors.

### Framework

* [**volcengine/verl**](https://github.com/volcengine/verl) (Apache 2.0,
  Bytedance Seed) — PPO / GRPO training engine. Pinned at `v0.4.0` and
  consumed as a pip dependency (not vendored); recipe-level extensions
  live under [`recipe/teleboost/`](recipe/teleboost/) and
  [`teleboost/`](teleboost/).

### Algorithms

Our reimplementations under
[`recipe/teleboost/algorithms/`](recipe/teleboost/algorithms/):

* **DanceGRPO** — Xue et al., *"DanceGRPO: Unleashing GRPO on Visual
  Generation"*, [arXiv:2505.07818](https://arxiv.org/abs/2505.07818)
* **Flow-GRPO** — Liu et al., *"Flow-GRPO: Training Flow Matching Models
  via Online RL"*, [arXiv:2505.05470](https://arxiv.org/abs/2505.05470)
* **GRPO-Guard** — Sun, Wang, et al., *"GRPO-Guard: Stable
  Diffusion-Style RL by Bias and Step-size Correction"*,
  [arXiv:2510.22319](https://arxiv.org/abs/2510.22319)
* **BGPO** — Liu, Liang, et al., *"Learning What to Trust: Bayesian
  Prior-Guided Optimization for Visual Generation"*,
  [arXiv:2511.18919](https://arxiv.org/abs/2511.18919)
* **VIPO** — Ni, Liang, et al., *"Seeing What Matters: Visual Preference
  Policy Optimization for Visual Generation"*,
  [arXiv:2511.18719](https://arxiv.org/abs/2511.18719)

### Actor models

* [**Wan-Video/Wan2.1**](https://github.com/Wan-Video/Wan2.1) and
  [**Wan-Video/Wan2.2**](https://github.com/Wan-Video/Wan2.2)
  (Apache 2.0; weights under additional Wan-Video community terms —
  check the upstream model cards before commercial use). Vendored under
  [`wan/`](wan/) with the original [`wan/LICENSE`](wan/LICENSE)
  preserved per Apache 2.0 §4(a). The FSDP / Ulysses-SP wrap in
  [`teleboost/patches/`](teleboost/patches/) is our addition.

### Reward models

* [**tgxs002/HPSv2**](https://github.com/tgxs002/HPSv2) (Apache 2.0) —
  Wu et al., [arXiv:2306.09341](https://arxiv.org/abs/2306.09341).
* [**LAION-AI/aesthetic-predictor**](https://github.com/LAION-AI/aesthetic-predictor)
  (MIT).
* [**princeton-vl/RAFT**](https://github.com/princeton-vl/RAFT)
  (BSD-3-Clause) — Teed & Deng,
  [arXiv:2003.12039](https://arxiv.org/abs/2003.12039).
* [**alibaba-pai/VideoCLIP-XL**](https://huggingface.co/alibaba-pai/VideoCLIP-XL)
  (license: see HF model card; verify before redistribution) —
  [arXiv:2410.00741](https://arxiv.org/abs/2410.00741).
* [**videophysics/videocon_physics**](https://huggingface.co/videophysics/videocon_physics)
  (license: see HF model card — **commonly CC-BY-NC; verify before
  commercial use**) —
  [arXiv:2406.03520](https://arxiv.org/abs/2406.03520).
* [**facebook/dinov2**](https://github.com/facebookresearch/dinov2)
  (Apache 2.0 for code; weights have separate terms — check upstream)
  — Oquab et al.,
  [arXiv:2304.07193](https://arxiv.org/abs/2304.07193). Used by VIPO
  for per-pixel allocation maps.

### Systems

* [**vllm-project/vllm**](https://github.com/vllm-project/vllm)
  (Apache 2.0) — Qwen reward worker rollout.
* [**Dao-AILab/flash-attention**](https://github.com/Dao-AILab/flash-attention)
  (BSD-3-Clause) — flash-attn 2.7.4.post1 used in actor + reward rollouts.
* [**flashinfer-ai/flashinfer**](https://github.com/flashinfer-ai/flashinfer)
  (Apache 2.0) — vLLM kernel backend.
