# TeleBoost — GRPO for video diffusion

A production RL training stack for Wan2.1 / Wan2.2 text-to-video diffusion,
built as a recipe on top of
[`volcengine/verl`](https://github.com/volcengine/verl). Six paper-pinned
algorithm variants (GRPO, DanceGRPO, Flow-GRPO, GRPO-Guard, BGPO, VIPO),
a CP gradient regression test that passes bit-exact at sp ∈ {1, 2, 4, 8},
and a four-model joint reward (aesthetic + RAFT + VideoCLIP + VideoPhy)
all out of the box.

verl is consumed as a plain pip dependency, not vendored. Wan-specific
behavior that doesn't belong in upstream verl (model loader, FSDP wrap,
Ulysses SP patches, checkpoint compatibility) lives under
[`teleboost/`](teleboost/) and is applied at import time.

## Algorithms

Each module in `recipe/teleboost/algorithms/` is a paper-faithful
translation; see the per-algorithm docstring for the equation pin.

| Algorithm | Paper | What it does |
|---|---|---|
| **GRPO** | [DeepSeekMath, arXiv 2402.03300](https://arxiv.org/abs/2402.03300) | Per-prompt z-score advantage (group baseline) |
| **DanceGRPO** | [arXiv 2505.07818](https://arxiv.org/abs/2505.07818) | GRPO for visual generation; SDE recast with σ_t = η constant |
| **Flow-GRPO** | [arXiv 2505.05470](https://arxiv.org/abs/2505.05470) | σ_t = η·√(t/(1−t)) form + sliding-window SDE |
| **GRPO-Guard** | [arXiv 2510.22319](https://arxiv.org/abs/2510.22319) | RatioNorm (Eq. 8) + grad-reweight δ (Eq. 12) |
| **BGPO** | [arXiv 2511.18919](https://arxiv.org/abs/2511.18919) | CRT reward rerange (Eq. 4) + RAS adaptive scaling (Eq. 2) |
| **VIPO** | [arXiv 2511.18719](https://arxiv.org/abs/2511.18719) | DINOv2 PCA → per-pixel allocation map → dense advantage |

Reward models / vendored components:

* **HPSv2** — [arXiv 2306.09341](https://arxiv.org/abs/2306.09341)
* **VideoCLIP-XL** — [arXiv 2410.00741](https://arxiv.org/abs/2410.00741)
* **RAFT (optical flow)** — [arXiv 2003.12039](https://arxiv.org/abs/2003.12039)
* **VideoPhy** — [arXiv 2406.03520](https://arxiv.org/abs/2406.03520)
* **LAION Aesthetic predictor** ([repo](https://github.com/LAION-AI/aesthetic-predictor))
* **DINOv2** — [arXiv 2304.07193](https://arxiv.org/abs/2304.07193) (used by VIPO)
* **Wan video diffusion** — [Wan-AI](https://github.com/Wan-Video) (actor backbone)

## Capability matrix

"Verified" = end-to-end training run completes with non-zero gradients and
a decreasing loss curve; not a claim of paper-SOTA reproduction.

| Dimension | Verified | Code present (untested) |
|---|---|---|
| Actor    | Wan2.2-T2V-A14B (`wan_version=wan22`), Wan2.1-T2V-1.3B (`wan_version=wan21`) | Wan2.2-I2V-A14B, Hunyuan, Mochi |
| Reward   | HPSv2, Qwen-VL-7B, 4-reward joint (aesthetic + raft + videoclip + videophy) | Qwen-VL-32B, custom callable |
| Algorithm | GRPO | GRPO-Guard, Flow-GRPO SDE path, GAE / RLOO (upstream) |
| Rollout  | Diffusion (actor), vLLM (Qwen reward) | sglang, hf, flowgrpo, mixgrpo |
| Sequence parallel | sp=1 / sp=2 / sp=8 (Wan22, CP grad bit-exact at fp32) | other Ulysses configs |
| Hardware | 4×H800 80 GB, 8×H800 80 GB | 8 GPU multi-host |

## Install

If you already have a verl-compatible Docker image (e.g.
`verlai/verl:vllm017.latest`), see [`INSTALL.md`](INSTALL.md) — three pip
commands plus an import check.

For a bare GPU host, see
[`docs/install_from_scratch.md`](docs/install_from_scratch.md) — full
recipe including the flash-attn wheel, hpsv2 packaging fixes, and known
gotchas.

## Prepare data

Every training row must carry both `context_path` (positive prompt umT5
embedding) and `context_null_path` (a single shared negative-prompt umT5
embedding for CFG). The dataset loader fails fast if `context_null_path`
is missing — without it CFG collapses to `(1+scale) * cond` and reward
variance vanishes (advantage=0, grad_norm=0).

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
* produces `processed_wan_prompt.json` with all three fields populated, plus
  per-row `context_<i>.npy` and a single shared `context_null.npy`;
* the negative prompt defaults to Wan's official Chinese template; override
  with `--negative_prompt "..."` if you need a different one.

The same `--wan_model_path` works for 1.3B and 14B (the script only uses
its umT5 checkpoint files).

## Train

The unified launcher `run_teleboost.sh` covers every algorithm variant
via env vars. Example for an 8-GPU 480×832 1000-step Wan2.2 run with
HPSv2 reward:

```bash
TELEBOOST_METHOD=baseline \
N_GPUS_PER_NODE=8 \
TOTAL_TRAINING_STEPS=1000 \
SAMPLING_STEPS=10 \
VIDEO_HEIGHT=480 VIDEO_WIDTH=832 NUM_FRAMES=49 \
INIT_SAME_NOISE=False \
TRAIN_FILE=/path/to/processed_wan_prompt.json \
TEST_FILE=/path/to/processed_wan_prompt.json \
WAN_MODEL_PATH=/path/to/Wan2.2-T2V-A14B \
WAN_VERSION=wan22 \
WAN_VAE_PATH=/path/to/Wan2.2-T2V-A14B/Wan2.1_VAE.pth \
REWARD_MODEL_PATH=/path/to/HPS_v2.1_compressed.pt \
bash recipe/teleboost/run_teleboost.sh
```

### Algorithm selector

| `TELEBOOST_METHOD` | Behavior | Required env vars (in addition to the core ones) |
|---|---|---|
| `baseline` (default) | plain GRPO | — |
| `bgpo` | GRPO + Bayesian-Prior reranging + RAS adaptive scaling | training rows must carry a `prior` field |
| `vipo` | GRPO + DINOv2 dense pixel-weight broadcast | `PIXEL_WEIGHT_MODEL_PATH=...` (default `facebook/dinov2-large`) |
| `bgpo_vipo` | both | both |
| `joint` | 4-reward joint (aesthetic + raft + videoclip + videophy) | `JOINT_AESTHETIC_CLIP_PATH`, `JOINT_AESTHETIC_MODEL_PATH`, `JOINT_RAFT_MODEL_PATH`, `JOINT_VIDEOCLIP_MODEL_PATH`, `JOINT_VIDEOPHY_MODEL_PATH` |

### Common knobs

| Env var | Default | Notes |
|---|---|---|
| `N_GPUS_PER_NODE` | `4` | per-node world size |
| `SAMPLING_STEPS` | `4` | denoising steps in rollout |
| `TOTAL_TRAINING_STEPS` | `2` | total optimizer steps — bump for real training |
| `VIDEO_HEIGHT`, `VIDEO_WIDTH`, `NUM_FRAMES` | `256`, `256`, `9` | resolution & frame count |
| `TRAIN_PROMPT_BSZ`, `N_RESP_PER_PROMPT` | `2`, `2` | batch shape (effective batch = bsz × n_resp) |
| `WAN_VERSION` | `wan21` | `wan22` for Wan2.2 dual-model A14B |
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
| `VAL_BEFORE_TRAIN=True` | run validation before step 0 |

Example — 8-GPU sp=8 BGPO+VIPO:

```bash
TELEBOOST_METHOD=bgpo_vipo N_GPUS_PER_NODE=8 SP_SIZE=8 \
SAMPLING_STEPS=10 INIT_SAME_NOISE=False \
TRAIN_FILE=... TEST_FILE=... WAN_MODEL_PATH=... REWARD_MODEL_PATH=... \
bash recipe/teleboost/run_teleboost.sh
```

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

## Tests

```bash
# Imports — must print 8/8 OK
python3 - <<'PY'
import importlib
mods = [
    "recipe.teleboost.main_teleboost",
    "recipe.teleboost.teleboost_ray_trainer",
    "recipe.teleboost.teleboost_fsdp_worker",
    "recipe.teleboost.dp_actor",
    "recipe.teleboost.unified_reward_worker",
    "teleboost.workers.reward_manager.dancegrpo",
    "teleboost.workers.rollout.diffusion_rollout",
    "teleboost.workers.sharding_manager.diffusion",
]
ok = sum(bool(importlib.import_module(m)) for m in mods)
print(f"{ok}/{len(mods)} pass")
PY

# CP grad reduce regression — should print "PASS rel_err=0" at fp32 sp=4 / sp=8
torchrun --nproc_per_node=4 tests/special_distributed/test_cp_grad_reduce.py
DTYPE=bf16 torchrun --nproc_per_node=4 tests/special_distributed/test_cp_grad_reduce.py
torchrun --nproc_per_node=8 tests/special_distributed/test_cp_grad_reduce.py
```

## Documentation

| File | Topic |
|---|---|
| [`INSTALL.md`](INSTALL.md) | Fast install path on a verl-ready Docker image |
| [`docs/install_from_scratch.md`](docs/install_from_scratch.md) | Bare-host install + every gotcha |
| [`requirements-pinned.txt`](requirements-pinned.txt) | Full pin file (every transitive dep) |

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

* GRPO ([DeepSeekMath, arXiv 2402.03300](https://arxiv.org/abs/2402.03300))
* DanceGRPO ([arXiv 2505.07818](https://arxiv.org/abs/2505.07818))
* Flow-GRPO ([arXiv 2505.05470](https://arxiv.org/abs/2505.05470))
* GRPO-Guard ([arXiv 2510.22319](https://arxiv.org/abs/2510.22319))
* BGPO ([arXiv 2511.18919](https://arxiv.org/abs/2511.18919))
* VIPO ([arXiv 2511.18719](https://arxiv.org/abs/2511.18719))

See each module's docstring for the equation pin.

**Models — actor backbones**

* [**Wan-Video/Wan2.1**](https://github.com/Wan-Video/Wan2.1) and
  Wan-Video/Wan2.2 — text-to-video diffusion actor. The `wan/`
  directory bundles their model code; the FSDP / Ulysses-SP wrap and
  CP-grad-reduce fix in [`teleboost/patches/`](teleboost/patches/) are
  our additions.

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
