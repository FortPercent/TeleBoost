# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.
"""DPO dataset base class for OSS users.

Subclass this and implement ``__getitem__`` to integrate your own data
source (lmdb / webdataset / huggingface datasets / preencoded latents on
disk / etc.) with teletron's training loop.

Schema contract — each ``__getitem__(idx)`` must return a dict shaped like:

    {
        "context":  Tensor[S_text, D_text],   # T5 / text-encoder output
        "chosen":   {
            "latents":           Tensor[C, T_c, H_c, W_c],   # VAE-encoded video
            "img_clip_feature":  Tensor[N_clip, D_clip],     # CLIP image feature
            "img_emb_y":         Tensor[C, T_c, H_c, W_c],   # reference frame latent
        },
        "rejected": {  # same keys; T_r / H_r / W_r MAY differ from chosen
            "latents":           Tensor[C, T_r, H_r, W_r],
            "img_clip_feature":  Tensor[N_clip, D_clip],
            "img_emb_y":         Tensor[C, T_r, H_r, W_r],
        },
    }

Notes
-----
* All tensors should be CPU and bf16 (or fp32 — teletron auto-casts).
* Batch dim is ADDED by the DataLoader collator; do NOT prepend B here.
* Chosen and rejected MAY have different temporal/spatial shapes. The
  training loop runs each branch through a separate forward pass
  (`_run_branch` in pretrain_dpo_i2v.py), so shape mismatch is supported
  by design — verified end-to-end with mismatched-shape FakeDataset
  (see tests/).
* If you do not need split-DPO's per-branch backward (you want a single
  preference loss instead), still return both branches and the framework
  handles the rest.

Minimal working subclass:

    class MyDPODataset(DPODatasetBase):
        def __init__(self, manifest_csv, vae, text_encoder, clip):
            self.rows = pd.read_csv(manifest_csv).to_dict("records")
            self.vae, self.text_encoder, self.clip = vae, text_encoder, clip

        def __len__(self): return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            chosen_video  = load_video(row["chosen_path"])
            reject_video  = load_video(row["rejected_path"])
            with torch.no_grad():
                ctx        = self.text_encoder(row["prompt"])
                ch_lat     = self.vae.encode(chosen_video)
                rj_lat     = self.vae.encode(reject_video)
                ch_clip    = self.clip(chosen_video[0])
                rj_clip    = self.clip(reject_video[0])
            return {
                "context": ctx,
                "chosen":   {"latents": ch_lat, "img_clip_feature": ch_clip,
                             "img_emb_y": ch_lat[:, :1].clone()},
                "rejected": {"latents": rj_lat, "img_clip_feature": rj_clip,
                             "img_emb_y": rj_lat[:, :1].clone()},
            }

Then register it:

    from teletron.datasets.build import DATASETS
    DATASETS.register_module(MyDPODataset)

And select it via your config-path:

    config = dict(dataset=dict(type="MyDPODataset", manifest_csv="...", ...))
"""
from __future__ import annotations
from typing import Mapping, Optional

import torch


class DPODatasetBase(torch.utils.data.Dataset):
    """Abstract base class — subclass and implement ``__getitem__``.

    See module docstring for the schema each ``__getitem__`` must return.
    """

    # Subclass interface ────────────────────────────────────────────────
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Mapping:
        raise NotImplementedError

    # Optional helper subclasses can call to validate output schema in tests.
    @staticmethod
    def _validate_item(item: Mapping, allow_mismatched_shapes: bool = True) -> None:
        """Lightweight schema check. Call from a unit test, not the training
        loop — this is O(item) every call and not free.

        Set ``allow_mismatched_shapes=False`` if you want to enforce that
        chosen and rejected have identical shapes (uncommon — most DPO
        setups allow per-branch lengths).
        """
        required_top = {"context", "chosen", "rejected"}
        missing = required_top - set(item.keys())
        if missing:
            raise ValueError(f"DPO item missing top-level keys: {missing}")
        for branch in ("chosen", "rejected"):
            sub = item[branch]
            if not isinstance(sub, Mapping):
                raise TypeError(f"item['{branch}'] must be a Mapping; got {type(sub)}")
            req = {"latents", "img_clip_feature", "img_emb_y"}
            miss = req - set(sub.keys())
            if miss:
                raise ValueError(f"item['{branch}'] missing keys: {miss}")
        if not allow_mismatched_shapes:
            ch_lat = item["chosen"]["latents"]
            rj_lat = item["rejected"]["latents"]
            if ch_lat.shape != rj_lat.shape:
                raise ValueError(
                    f"chosen.latents shape {ch_lat.shape} != "
                    f"rejected.latents shape {rj_lat.shape}; either pad to a common "
                    f"shape in your dataset or set allow_mismatched_shapes=True."
                )
