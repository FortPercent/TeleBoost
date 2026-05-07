"""Smoke-test override of wan_dpo.py: 1.3B T2V + FakeDPODataset.

Runs the production training graph end-to-end with no real data and a
tiny 1.3B random-init DiT, so a 2-iter loop can validate parallelism /
forward / backward / DPO loss wiring without needing CSV data, video
mounts, CLIP weights, or pretrained DiT checkpoints.

Override entry: pass `--config-path config.wan_dpo_smoke.config`.
"""
import copy
import os

from .wan_dpo import config as _base

config = copy.deepcopy(_base)

# === FakeDPODataset — pre-encoded random pairs, skips CSV / video / VAE / T5 ===
config["dataset"] = dict(type="FakeDPODataset")

# === Shrink DiT 14B -> 1.3B T2V ===
config["model_config"]["dit"]["config"].update(
    dict(
        has_image_input=False,
        in_dim=16,
        dim=1536,
        ffn_dim=8960,
        num_heads=12,
        num_layers=30,
    )
)
config["model_config"]["dit"]["train"]["extra_inputs"] = []  # t2v: no input_image

# === Point encoders at the downloaded 1.3B weights ===
_W = os.environ.get("WAN13B_DIR", "/gfs/space/chatrl/users/wuxn5/wan_ckpt/Wan2.1-T2V-1.3B")
config["model_config"]["encoder"]["vae"]["path"] = f"{_W}/Wan2.1_VAE.pth"
config["model_config"]["encoder"]["text_encoder"]["path"] = f"{_W}/models_t5_umt5-xxl-enc-bf16.pth"
config["model_config"]["encoder"]["text_encoder"]["tokenizer_path"] = f"{_W}/google/umt5-xxl"

# === T2V: drop CLIP image_encoder + img_emb_y schema ===
config["model_config"]["encoder"].pop("image_encoder", None)
config["model_config"]["encoder"]["encoder_schema"] = ["context", "latents"]
