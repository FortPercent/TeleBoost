# TeleaiEncoder WORK_FN Reference

This document maps each WORK_FN key to its underlying function, required
batch fields, and output fields. It is based on:
- `teletron/models/teleai/teleai_encoder.py`
- `teletron/models/teleai/teleai_encoder_utils.py`

## Schema Mapping

`TeleaiEncoder.get_output_schema()` is defined by
`model_config.encoder.encoder_schema`. The encoder outputs only the keys
listed in that schema (default: `["context", "latents"]`).

Each schema key maps to a WORK_FN entry:

| Schema key | Function | Output field |
| --- | --- | --- |
| `context` | `get_context` | `context` tensor |
| `prompt_emb` | `get_context` | `context` tensor (alias output) |
| `unprompt_emb` | `get_unprompt_emb` | `context` tensor (negative prompt) |
| `img_clip_feature` | `get_img_clip_feature` | `clip_context` tensor |
| `img_emb_y` | `get_img_emb_y` | `y` tensor |
| `latents` | `get_latents` | `latents` tensor |
| `noise` | `get_noise` | `noise` tensor |
| `fake_latents` | `get_fake_latents` | `fake_latents` tensor |
| `depth_latents` | `get_depth_latents` | `depth_latents` tensor |

Notes:
- Output fields are the return values from the work functions. The final
  encoded output uses the schema key as the dict key.
- `prompt_emb` uses `get_context`, so it returns a `context` embedding but
  is stored under `prompt_emb` in the output dict.

## Required Batch Fields

The work functions consume fields from `batch` (or `raw_batch["chosen"]` /
`raw_batch["rejected"]` for DPO). Most functions assume:
`batch["images"]` is shaped as `B x T x C x H x W`.

Per WORK_FN:

| WORK_FN key | Required fields | Optional fields | Behavior summary |
| --- | --- | --- | --- |
| `context` | `struct_prompt` | - | Encodes prompt via prompter and returns `context`. |
| `prompt_emb` | `struct_prompt` | - | Same as `context`, returned under `prompt_emb`. |
| `unprompt_emb` | none | - | Uses `args.negative_prompt` and `args.micro_batch_size`. |
| `img_clip_feature` | `images`, `raw_first_image` | - | Uses first image to encode CLIP features. |
| `img_emb_y` | `images` | `ref_images`, `ref_mask`, `raw_first_image` | Encodes image latents and mask into `y`. |
| `latents` | `images` | - | VAE-encodes video to latents. |
| `noise` | `latents` or `images` | - | If `latents` exists, returns `randn_like(latents)`, else shape from `images`. |
| `fake_latents` | `images` | - | Low-res VAE encode + upsample to latent size. |
| `depth_latents` | `images` | - | Depth estimation -> VAE encode depth latents. |

Additional details:
- `img_clip_feature`:
  - Only `raw_first_image` path is implemented.
  - `raw_last_image` and `ref_images` are explicitly `NotImplemented`.
- `img_emb_y`:
  - If `ref_images` is provided, uses `ref_mask` and VAE encodes `ref_images`.
  - If `raw_first_image` is provided, builds a mask and VAE encodes the
    first frame.
  - If neither is provided, this path will fail.

## DPO Batch Behavior

If `raw_batch` contains both `chosen` and `rejected`, the encoder uses
`_encode_dpo`:

- Shared keys: `context`, `prompt_emb`, `unprompt_emb` are computed once
  from `raw_batch["chosen"]` and stored at the top level.
- All other schema keys are computed separately for `chosen` and `rejected`
  and returned under:
  - `out["chosen"]`
  - `out["rejected"]`

## Output Structure Examples

Single batch (non-DPO):
```
{
  "context": <tensor>,
  "latents": <tensor>,
  ...
}
```

DPO batch:
```
{
  "context": <tensor>,
  "chosen": {"latents": <tensor>, ...},
  "rejected": {"latents": <tensor>, ...}
}
```

