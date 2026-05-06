# Dataset preprocess plan

## Why this is needed

Training requires per-prompt umT5-XXL text embeddings (`.npy`) plus a
JSON manifest that lists each prompt's `caption`, `context_path`
(positive prompt embedding), and `context_null_path` (shared
negative-prompt embedding for CFG).

If you obtain a `processed_wan_prompt.json` from someone else's pipeline
and the `caption` field is garbled (single-char artifacts like `"["` or
`"{"` from a parser bug during file copy), HPS-v2 will compute the
reward against the literal `"["` text and return ~0.  The training loop
runs end-to-end, but the reward signal is meaningless.

## What to do

Run the bundled preprocess against a real prompts file:

```bash
python data_preprocess/prepare_wan_data.py \
    --input <prompts_file>            # one prompt per line, or list-of-{caption,...} JSON
    --output_dir <output_dir>          # where context_*.npy + processed_wan_prompt.json land
    --wan_model_path <wan_ckpt_dir>    # used for the T5 encoder
```

`prepare_wan_data.py` is idempotent: regenerating already-good rows is a
no-op, so it's safe to re-run if the previous attempt was interrupted.

Outputs (under `--output_dir`):

* `context_<i>.npy`              — umT5-XXL embedding of each prompt
* `context_null.npy`             — embedding of the negative prompt (shared)
* `processed_wan_prompt.json`    — per-row `{caption, context_path, context_null_path}`

## Hardware / time

* Single GPU is enough — the T5 encoder loads in bf16 (~10 GB).
* ~50 prompts: a few seconds.  ~2k prompts: a few minutes.
* Choose an output path on persistent storage; ephemeral / per-pod
  `/workspace`-style mounts will lose the embeddings on container
  restart.

## Validation after regen

Re-run a default training against the new dataset; `train/rewards` should
land in HPS-v2's typical ~0.15–0.30 range for random Wan2.2 outputs.

```bash
TRAIN_FILE=<output_dir>/processed_wan_prompt.json \
TEST_FILE=$TRAIN_FILE \
WAN_MODEL_PATH=<wan_ckpt_dir> \
REWARD_MODEL_PATH=<hps_v2.1_compressed.pt> \
bash recipe/teleboost/run_teleboost.sh data.prompt_key=caption
```

If `train/rewards` is still 0, two things to check:

* `data.prompt_key` — the launcher defaults to `prompt`; the JSON
  written by `prepare_wan_data.py` uses `caption`, so pass
  `data.prompt_key=caption` as a Hydra override.
* `reward_model.normalize` — defaults to `true`, which z-scores rewards
  to mean 0 across the batch.  The normalized metric will read 0 even
  when raw HPS scores are non-zero; pass
  `reward_model.normalize=false` to inspect raw rewards.

## Why this is a doc, not a script

The non-obvious parts are:

1. **Which prompt list to use** — depends on the research goal
   (istock-style natural prompts vs. curated hard prompts vs. domain
   data).
2. **Output location** — depends on storage quota and persistence
   guarantees.
3. **Quality check** — post-regen run + visual VAE-decode dump that
   verifies the prompt → video alignment is real.

These belong to the operator, not encoded in the script.
