# TeleBoost — DanceGRPO for Wan video diffusion

A video-generation RL training stack built on top of upstream
[`volcengine/verl`](https://github.com/volcengine/verl) v0.4.0. The actor is a
Wan2.1 / Wan2.2 text-to-video diffusion backbone; the reward signal can come
from HPSv2, Qwen-VL, or a four-model joint setup (aesthetic + RAFT + VideoCLIP
+ videophy). The base RL algorithm is GRPO, with optional GRPO-Guard and
flow-grpo extensions.

This repo is the *recipe* + Wan integration; verl itself is consumed as a
plain pip dependency, not vendored. Wan-specific behaviour that doesn't
belong in upstream verl (model loader, FSDP wrap, Ulysses SP patches,
checkpoint compatibility) lives under [`teleboost/`](teleboost/) and is
applied at import time.

---

## 1. Capability matrix

| Dimension | Verified | Code present (untested) |
|---|---|---|
| Actor    | Wan2.2-T2V-A14B (`wan_version=wan22`), Wan2.1-T2V-1.3B (`wan_version=wan21`) | Wan2.2-I2V-A14B, Hunyuan, Mochi |
| Reward   | HPSv2, Qwen-VL-7B, 4-reward joint (aesthetic + raft + videoclip + videophy) | Qwen-VL-32B, custom callable |
| Algorithm | GRPO | GRPO-Guard, flow-grpo SDE path, GAE / RLOO (upstream) |
| Rollout  | Diffusion (actor), vLLM (Qwen reward) | sglang, hf, flowgrpo, mixgrpo |
| Sequence parallel | sp=1 / sp=2 / sp=8 (Wan22, smoke + CP grad bit-exact at fp32) | other Ulysses configs |
| Hardware | 4×H800 80 GB, 8×H800 80 GB | 8 GPU multi-host |

---

## 2. Quickstart

If you already have a verl-compatible Docker image (e.g. `verlai/verl:vllm017.latest`):
see [`INSTALL.md`](INSTALL.md) — three pip commands plus an import-smoke check.

If you're starting from a bare GPU host:
see [`docs/install_from_scratch.md`](docs/install_from_scratch.md) — full
recipe including the flash-attn wheel, hpsv2 packaging fixes, and the
gotchas you'll otherwise hit.

### 2.1 Prepare data

Every training row must carry both `context_path` (positive prompt umT5
embedding) and `context_null_path` (a single shared negative-prompt umT5
embedding for CFG). The dataset loader fails fast if `context_null_path`
is missing — without it CFG collapses to `(1+scale) * cond` and reward
variance vanishes (advantage=0, grad_norm=0; documented training-killer
in the pre-X3 fork).

One unified, idempotent prep script handles both flat prompt lists and
existing JSON layouts:

```bash
# from a plain prompts.txt (one prompt per line)
python data_preprocess/prepare_wan_data.py \
  --input prompts/hard_50.txt \
  --output_dir data/processed/ \
  --wan_model_path /path/to/Wan2.1-T2V-1.3B

# patching an existing JSON that's missing context_null_path
python data_preprocess/prepare_wan_data.py \
  --input data/processed/processed_wan_prompt.json \
  --output_dir data/processed/ \
  --wan_model_path /path/to/Wan2.1-T2V-1.3B
```

The script:
- accepts `.txt` (one prompt per line) or `.json` (`[{"caption": ...}, ...]`);
- only loads the umT5 encoder if something actually needs encoding;
- per row, skips T5 if `context_path` already points at an existing `.npy`;
- produces `processed_wan_prompt.json` with all three fields populated, plus
  per-row `context_<i>.npy` and a single shared `context_null.npy`;
- the negative prompt defaults to Wan's official Chinese template; override
  with `--negative_prompt "..."` if you need a different one.

Same `--wan_model_path` works for 1.3B and 14B (the script only uses its umT5
checkpoint files).

### 2.2 Run a smoke

After install + checkpoint downloads + data prep, override the paths and run:

```bash
TRAIN_FILE=/path/to/processed_wan_prompt.json \
TEST_FILE=/path/to/processed_wan_prompt.json \
CKPTS_DIR=/tmp/dancegrpo_smoke_ckpt \
bash recipe/dancegrpo/run_dancegrpo_single_4gpu_smoke.sh
```

Smoke variants (all run a complete rollout + reward + actor.update + save loop in 2 steps):

| Script | Actor | Reward | World × SP |
|---|---|---|---|
| `run_dancegrpo_1p3B_4gpu_smoke.sh` | Wan2.1-T2V-1.3B | HPSv2 | 4 × 1 |
| `run_dancegrpo_1p3B_qwen_4gpu_smoke.sh` | Wan2.1-T2V-1.3B | Qwen-VL-7B | 4 × 1 |
| `run_dancegrpo_1p3B_joint_4gpu_smoke.sh` | Wan2.1-T2V-1.3B | 4-reward joint | 4 × 1 |
| `run_dancegrpo_single_4gpu_smoke.sh` | Wan2.2-T2V-A14B | HPSv2 | 4 × 1 |
| `run_dancegrpo_single_4gpu_smoke_sp2.sh` | Wan2.2-T2V-A14B | HPSv2 | 4 × 2 |
| `run_dancegrpo_single_4gpu_joint_smoke.sh` | Wan2.2-T2V-A14B | 4-reward joint | 4 × 1 |
| `run_dancegrpo_single_4gpu_qwen_smoke.sh` | Wan2.2-T2V-A14B | Qwen-VL-7B | 4 × 1 |
| `run_dancegrpo_single_8gpu_smoke_sp8.sh` | Wan2.2-T2V-A14B | HPSv2 | 8 × 8 |

Replace the actor / VAE / reward paths inside each script for your
environment.

### 2.3 Production training

Production scripts (8 GPUs, 480×832 resolution, full denoising + 1000 training steps):

```
recipe/dancegrpo/run_dancegrpo_single.sh        # single HPSv2 reward
recipe/dancegrpo/run_dancegrpo_qwen.sh          # Qwen-VL reward
recipe/dancegrpo/run_dancegrpo_joint.sh         # 4-reward joint
```

---

## 3. Repository layout

```
recipe/dancegrpo/                     Recipe (entry point + workers + scripts)
├── main_dancegrpo.py                     Hydra entry; spawns Ray workers
├── dancegrpo_ray_trainer.py              RayPPOTrainer subclass
├── dp_actor.py                           DataParallelPPOActor subclass
├── dancegrpo_fsdp_worker.py              7 reward workers + DiffusionActorRolloutRefWorker subclass
├── unified_reward_worker.py              plugin-style reward worker
├── reward_models/                        reward registry + 5 plugins + composite + dynamic_joint
├── config/dancegrpo_trainer.yaml         Hydra config (grpo_guard, flow_grpo, joint.* fields)
└── run_dancegrpo_*.sh                    smoke + production launch scripts

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
prompts/                              prompt lists for preprocess
docs/                                 docs (see §4)
tests/special_distributed/            distributed regression tests (CP grad reduce)
```

`teleboost/__init__.py` runs `teleboost.patches.apply()` at import time, so
any process that does `import teleboost` (or imports anything under
`recipe.dancegrpo.*`, which imports teleboost) ends up with the patches
applied automatically.

---

## 4. Documentation

| File | Topic |
|---|---|
| [`INSTALL.md`](INSTALL.md) | Fast install path on a verl-ready Docker image |
| [`docs/install_from_scratch.md`](docs/install_from_scratch.md) | Bare-host install + every gotcha |
| [`requirements-pinned.txt`](requirements-pinned.txt) | Full pin file (every transitive dep) |

---

## 5. Tests

```bash
# Imports — must print 8/8 OK
python3 - <<'PY'
import importlib
mods = [
    "recipe.dancegrpo.main_dancegrpo",
    "recipe.dancegrpo.dancegrpo_ray_trainer",
    "recipe.dancegrpo.dancegrpo_fsdp_worker",
    "recipe.dancegrpo.dp_actor",
    "recipe.dancegrpo.unified_reward_worker",
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

---

## 6. License & acknowledgments

Apache 2.0 (see [`LICENSE`](LICENSE) and [`Notice.txt`](Notice.txt)).

Built on top of:
- [volcengine/verl](https://github.com/volcengine/verl) — RLHF framework
- [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) and Wan-Video/Wan2.2 — diffusion backbones
- [tgxs002/HPSv2](https://github.com/tgxs002/HPSv2), [alibaba-pai/VideoCLIP-XL](https://huggingface.co/alibaba-pai/VideoCLIP-XL), [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor), [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT), [videophysics/videocon_physics](https://huggingface.co/videophysics/videocon_physics) — reward models

---

## 7. FLUX training notes (legacy)

Earlier iterations of this codebase trained FLUX, not Wan. The notes below
were valid for that path; they're recorded here for reference but do not
apply to the current Wan-on-verl mainline.

1. We set the inference batch size to 1 because we observed differences in probability outputs when it exceeds the training batch size.
2. A stronger SFT stage can suppress exploration during the GRPO phase.
3. For extreme cases (same prompt + same initial noise + reward can't distinguish), try varying initial noise within a prompt.
4. Extended training (larger `max_train_steps`) may not improve visualization quality due to reward model limits (HPS-v2.1 not optimized for FLUX). EMA support is planned.
