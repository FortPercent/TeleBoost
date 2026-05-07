"""Smoke-test override of wan_dpo.py: 1.3B T2V + 10-pair real-video DPO.

Runs the production training graph end-to-end against a tiny 10-pair
DPO CSV built from /gfs/.../paired_videos. Uses 1.3B random-init DiT
(via ParallelTeleaiModel) so no DiT checkpoint is required; VAE / T5
load from the downloaded Wan2.1-T2V-1.3B; CLIP image-encoder is dropped
(t2v doesn't need it).

Override entry: pass `--config-path config.wan_dpo_smoke.config`.
"""
import copy
import os

from .wan_dpo import config as _base

config = copy.deepcopy(_base)

# === 10-pair real-data CSV (kling vs hailuo per paired_videos dir) ===
_PAIRS_CSV = os.environ.get(
    "DPO_SMOKE_CSV",
    "/gfs/space/chatrl/users/wuxn5/dpo_smoke/pairs_10.csv",
)
config["dataset"]["dataset_base_path"] = ""
config["dataset"]["dataset_metadata_path"] = _PAIRS_CSV
config["dataset"]["data_path_list"] = [_PAIRS_CSV]
# CSV columns are positive_video_path / negative_video_path (not chosen / rejected)
config["dataset"]["chosen_video_key"] = "positive_video_path"
config["dataset"]["rejected_video_key"] = "negative_video_path"
config["dataset"]["dataset_repeat"] = 1
# 10-pair smoke at lower res to keep encoder + 2-iter forward fast
config["dataset"]["height"] = 480
config["dataset"]["width"] = 832
config["dataset"]["num_frames"] = 49

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
