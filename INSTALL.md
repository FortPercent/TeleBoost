# TeleBoost Installation

TeleBoost is a video-generation RL training stack on top of upstream
[`volcengine/verl`](https://github.com/volcengine/verl) v0.4.0.

The repo is structured as:
- `recipe/dancegrpo/` — training recipe (PPO/GRPO entrypoint, FSDP worker,
  reward managers); imports `verl.*` from upstream + `teleboost.*` for
  TeleBoost-specific extensions.
- `wan/` — Wan2.1 / Wan2.2 video generation model code.
- `teleboost/` — extensions on top of upstream verl:
  - `teleboost.models.*` / `teleboost.workers.*` — symbols upstream verl
    doesn't ship (VideoCLIP_XL, Videophy, wan transformer adapters,
    diffusion rollout / sharding manager, etc.).
  - `teleboost.patches.*` — runtime monkey-patches that bridge between the
    upstream verl namespace and TeleBoost-specific behaviour (cp grad
    fix, backported helpers like `marked_timer`, `simple_timer`,
    `ProfilerConfig`, `WorkerProfilerExtension`, `get_device_id`,
    `get_nccl_backend`, `convert_weight_keys`, `register`).
  - Imported once in any entrypoint via `import teleboost`, which
    auto-runs `teleboost.patches.apply()` (idempotent).

## Prerequisites

- 8 × A100 / H100 / H800 (cuda 12.4)
- Python 3.10+
- A pre-built environment with the heavy native deps:
  - torch 2.6.0 + cu124
  - vllm 0.8.x
  - flash-attn 2.7.4.post1
  - ray 2.43.x
  - flash-attn / cudnn / nccl matching the above

The recommended way is the `verlai/verl:vllm017.latest` Docker image (or any
verl-compatible image with the above pre-installed). See
[the upstream Docker docs](https://hub.docker.com/r/verlai/verl).

## Install steps

```bash
# 1. clone
git clone https://github.com/FortPercent/TeleBoost.git
cd TeleBoost
git checkout import-verl    # or whichever branch/tag you target

# 2. install non-verl Python deps (will not touch torch/vllm/flash-attn etc.)
pip install -r requirements.txt

# 3. install upstream verl WITHOUT pulling its deps (those are managed above
#    and would otherwise upgrade torch/cuda/ray and break vllm/flash-attn)
pip install --no-deps -r requirements-verl.txt

# 4. install teleboost itself (current repo) without re-resolving deps
pip install --no-deps -e .
```

## Smoke test

```bash
# Import sanity: must print 8/8 OK with no traceback.
python3 - <<'PY'
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
ok = 0
for m in mods:
    try:
        __import__(m); ok += 1; print(f"OK   {m}")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {e}")
print(f"{ok}/{len(mods)} pass")
PY

# Optional: cp grad fix regression test (8 GPUs, ~30 sec)
torchrun --nproc_per_node=4 tests/special_distributed/test_cp_grad_reduce.py
```

## Run training

```bash
# 5-step smoke on 8 GPUs (single reward, wan2.2 actor)
bash recipe/dancegrpo/run_dancegrpo_single_8gpu_5step.sh
```

Replace the data / model paths inside the script for your environment.

## How patches work

When *any* code does `import teleboost`, `teleboost/__init__.py` immediately
calls `teleboost.patches.apply()` (idempotent). This injects
TeleBoost-specific symbols and overrides into the live `verl.utils.*` /
`verl.workers.*` modules, so the rest of the codebase can keep using the
canonical `from verl.X import Y` style.

To add a new patch: drop a module under `teleboost/patches/` exposing
`apply()`, then call it from `teleboost/patches/__init__.py:apply`. Document
the upstream symbol it adds / overrides at the top of the file.
