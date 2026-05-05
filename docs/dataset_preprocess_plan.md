# Dataset preprocess plan (action item, not yet executed)

## Why we need this

The 14B-Wan rl_embeddings dataset on the cluster
(`/gfs/platform/public/infra/qrl760/Dance_GRPO/Dancegrpo/data/14B/rl_embeddings/processed_wan_prompt.json`)
has **corrupted captions** — verified 2026-05-05:

```python
[0] caption = '['
[1] caption = '{'
[5] caption = '},'
[10] caption = '},'
```

The `.npy` umT5-XXL embeddings (`context_*.npy`, `context_null.npy`)
referenced by the JSON **are valid** — they were generated from real
prompts before the JSON's caption field was corrupted (likely a parser
bug during a copy/migration step).

Effect on training:
- **Smoke (this PR)**: training loop runs end-to-end (rollout → reward →
  advantage → actor update → checkpoint), but `train/rewards = 0` because
  HPS-v2 rewards a video against the literal caption "[" / "{" → 0 score.
- **Real training**: cannot run.  Reward signal is meaningless.

## What to do

Either:

### Option 1 (cheap, ~5 min): regenerate just the JSON's caption field

If we can recover the original prompts (the cluster's `/gfs/.../prompts/`
has `mini_test.txt`, `istock_2000.txt`, `istock_5w.txt`), and the order
of `context_*.npy` matches one of those prompt lists, we can rewrite the
JSON without re-running umT5.

Risk: we need to *prove* the order matches.  No metadata records which
prompt list was used to generate the existing `.npy` files.

### Option 2 (robust, ~30 min on 8 GPU): regenerate everything

Pick a prompt source (e.g. `istock_2000.txt`) and run the existing
preprocess pipeline:

```bash
cd /workspace/Dancegrpo
python data_preprocess/prepare_wan_data.py \
    --input /gfs/platform/public/infra/wxe/Dance-grpo/prompts/istock_2000.txt \
    --output_dir /gfs/<your-user>/teleboost/data/wan22_2k/rl_embeddings \
    --wan_model_path /gfs/platform/public/infra/qrl760/Dance_GRPO/models/Wan2.2-T2V-A14B
```

`prepare_wan_data.py` is idempotent (per its docstring): regenerating
already-good rows is a no-op, so safe to run multiple times.

Outputs:
- `context_<i>.npy` per prompt (umT5-XXL forward, ~50 MB each on 14B?)
- `context_null.npy` (one negative-prompt encode, shared)
- `processed_wan_prompt.json` with **valid** `caption` + `context_path`
  + `context_null_path` for every row

## Recommended path

**Option 2.**  The output is a clean dataset under our own user dir,
provenance is unambiguous (prompts file → embeddings → JSON), and the
preprocess is idempotent so it costs nothing to re-run.

## Hard pre-requisites before running

- Wan2.2 ckpt: `/gfs/platform/public/infra/qrl760/Dance_GRPO/models/Wan2.2-T2V-A14B/`
  (verified present 2026-05-05; needs ≥1 GPU to load `models_t5_umt5-xxl-enc-bf16.pth`).
- T5 encoder loads bf16 → ~10GB GPU RAM; single H800 plenty.
- Output dir: pick a path under `/gfs/space/chatrl/users/<you>/...` so
  it survives the per-pod `/workspace/` ephemerality.

## Smoke validation after regen

Re-run the existing baseline smoke against the new dataset; `train/rewards`
should now be a non-zero per-prompt HPS score (typical range 0.2–0.4 for
random Wan2.2 outputs against istock prompts).

```bash
TRAIN_FILE=/gfs/.../wan22_2k/rl_embeddings/processed_wan_prompt.json \
TEST_FILE=$TRAIN_FILE \
... (rest of smoke env from INSTALL.md) \
bash recipe/teleboost/run_teleboost_smoke.sh
```

If `train/rewards` is still 0, it means the dataset captions still don't
flow into HPS — investigate `data.prompt_key` in the smoke (currently
`prompt`, but new JSON's key is `caption`).

## Why this is in a doc not a script

Doing the regen is straightforward (one command).  The non-obvious parts
are:

1. **Which prompt list to use** (research call — istock_2000 vs the
   3w/5w sets, vs a project-specific prompt curation).
2. **Output location** (depends on user identity / quota).
3. **Quality check** (post-regen smoke verifies HPS gives non-zero,
   and visual VAE-decode dump verifies the prompt → video alignment is
   real).

These belong to the operator running the preprocess, not encoded into
the smoke script.
