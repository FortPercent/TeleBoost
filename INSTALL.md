# TeleBoost Installation

TeleBoost is a video-generation RL training stack on top of upstream
[`volcengine/verl`](https://github.com/volcengine/verl) v0.4.0.

The repo is structured as:
- `recipe/teleboost/` — training recipe (PPO/GRPO entrypoint, FSDP worker,
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
    "recipe.teleboost.main_teleboost",
    "recipe.teleboost.teleboost_ray_trainer",
    "recipe.teleboost.teleboost_fsdp_worker",
    "recipe.teleboost.dp_actor",
    "recipe.teleboost.unified_reward_worker",
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

The unified entrypoint is `recipe/teleboost/run_teleboost.sh`.
Required env vars:

```bash
export TRAIN_FILE=/path/to/processed_wan_prompt.json     # rl_embeddings JSON
export TEST_FILE=$TRAIN_FILE                              # can reuse train file
export WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B            # or Wan2.2-T2V-A14B
export REWARD_MODEL_PATH=/path/to/HPS_v2.1_compressed.pt  # for non-joint runs

bash recipe/teleboost/run_teleboost.sh
```

Optional knobs (defaults shown): `TELEBOOST_METHOD=baseline`, `WAN_VERSION=wan21`,
`N_GPUS_PER_NODE=4`, `TRAIN_PROMPT_BSZ=2`, `PPO_MINI_BATCH_SIZE=2`,
`TOTAL_TRAINING_STEPS=2`.  See script header for full list.

## Known gotchas (verified 2026-05-05 on 8×H800)

These are real failures we hit during fresh-box install / end-to-end runs.  The
Dockerfile already encodes the fixes; if you install by hand, watch for them.

1. **`pip install verl` pulls the wrong version.**  The latest PyPI verl
   (0.7.x) has dropped `RewardModelWorker` and several worker symbols this
   repo depends on.  Install the pinned v0.4.0 instead:
   ```
   pip install --no-deps -r requirements-verl.txt
   ```

2. **transformers 5.x breaks peft 0.17.**  peft tries to import
   `HybridCache` which only exists in transformers 4.x.  Pin
   `"transformers>=4.45,<5.0"`.

3. **HPSv2 PyPI wheel is missing the BPE vocab.**  After
   `pip install hpsv2==1.2.0`, drop the vocab file in:
   ```
   curl -fsSL -o $(python -c "import hpsv2.src.open_clip as m,os; print(os.path.dirname(m.__file__))")/bpe_simple_vocab_16e6.txt.gz \
     https://raw.githubusercontent.com/tgxs002/HPSv2/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz
   ```
   The Dockerfile does this for you.

4. **`torch._dynamo` corrupts the inductor cache during VAE decode under
   FSDP × 8 ranks.**  First-run JSONDecodeError in
   ``codecache._read``.  Fix: run with `TORCH_COMPILE_DISABLE=1`.  The
   Dockerfile sets this in the runtime ENV; bare-metal installs need it
   exported.

5. **`PPO_MINI_BATCH_SIZE` must be ≥ `n_gpus_per_node` after world-size
   normalization.**  verl normalizes the mini-batch by world size with
   floor division.  The launcher now defaults
   `PPO_MINI_BATCH_SIZE=${n_gpus}` so this scales automatically; if you
   override, keep it ≥ `n_gpus`.

6. **`tokenizer_subpath` (recipe-level), not `tokenizer_path` (verl).**
   verl's `hf_model.yaml` exposes `actor_rollout_ref.model.tokenizer_path`
   (a full HF tokenizer path, default `null` = use the model dir).
   That is *not* the same as our recipe-level
   `actor_rollout_ref.tokenizer_subpath` (a relative subpath joined onto
   the model dir, default `google/umt5-xxl` for Wan).  The two have
   different semantics; we keep them as different keys so verl's null
   default cannot accidentally clobber the recipe value.

7. **`actor_rollout_ref.actor.sigma_form=flow_grpo` needs the σ=1
   substitution.**  The Flow-GRPO formula `σ_t = √(σ/(1−σ))·η` has a
   pole at σ=1 (start of Wan/SD3 schedule).  Mirror upstream
   `sd3_sde_with_logprob.py`: replace σ=1 with σ_next.  This is
   already in `algorithms/sigma_schedule.py` and pinned by
   `tests/test_sigma_schedule.py::test_flow_grpo_sigma_one_edge_case_no_nan`.

## How patches work

When *any* code does `import teleboost`, `teleboost/__init__.py` immediately
calls `teleboost.patches.apply()` (idempotent). This injects
TeleBoost-specific symbols and overrides into the live `verl.utils.*` /
`verl.workers.*` modules, so the rest of the codebase can keep using the
canonical `from verl.X import Y` style.

To add a new patch: drop a module under `teleboost/patches/` exposing
`apply()`, then call it from `teleboost/patches/__init__.py:apply`. Document
the upstream symbol it adds / overrides at the top of the file.
