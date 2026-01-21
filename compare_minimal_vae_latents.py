#!/usr/bin/env python
# Usage:
#   python compare_minimal_vae_latents.py \
#     --a /tmp/min_vae_latents.pt \
#     --b /tmp/min_vae_latents_venv.pt
import argparse
import hashlib
from pathlib import Path

import torch


def _sha256_tensor_bytes(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def _describe_payload(payload):
    latents = payload.get("latents")
    if not torch.is_tensor(latents):
        return {"latents": None}
    return {
        "latents_shape": tuple(latents.shape),
        "latents_dtype": str(latents.dtype),
        "latents_sha256": _sha256_tensor_bytes(latents),
        "input_shape": payload.get("input_shape"),
        "input_dtype": payload.get("input_dtype"),
        "tiler_kwargs": payload.get("tiler_kwargs"),
        "vae_path": payload.get("vae_path"),
        "vae_weight_sha256": payload.get("vae_weight_sha256"),
        "vae_state_sha256": payload.get("vae_state_sha256"),
    }


def _compare_tensors(a, b, rtol, atol):
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        raise RuntimeError("both latents must be tensors")
    if a.shape != b.shape:
        raise RuntimeError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (a.float() - b.float()).abs()
    return {
        "allclose": bool(torch.allclose(a, b, rtol=rtol, atol=atol)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two minimal VAE latent dumps.")
    parser.add_argument("--a", required=True, help="First .pt file.")
    parser.add_argument("--b", required=True, help="Second .pt file.")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()

    path_a = Path(args.a)
    path_b = Path(args.b)
    if not path_a.exists():
        raise RuntimeError(f"file not found: {path_a}")
    if not path_b.exists():
        raise RuntimeError(f"file not found: {path_b}")

    payload_a = torch.load(path_a, map_location="cpu")
    payload_b = torch.load(path_b, map_location="cpu")

    info_a = _describe_payload(payload_a)
    info_b = _describe_payload(payload_b)
    print(f"[a] info={info_a}")
    print(f"[b] info={info_b}")

    latents_a = payload_a.get("latents")
    latents_b = payload_b.get("latents")
    result = _compare_tensors(latents_a, latents_b, args.rtol, args.atol)
    print(f"[compare] allclose={result['allclose']} max_abs={result['max_abs']} mean_abs={result['mean_abs']}")


if __name__ == "__main__":
    main()
