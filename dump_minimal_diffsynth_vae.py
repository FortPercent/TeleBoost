#!/usr/bin/env python
# Usage:
#   python dump_minimal_diffsynth_vae.py --vae-path /path/to/Wan2.1_VAE.pth \
#     --out /tmp/min_vae_latents.pt --device cuda \
#     --vae-module teletron.models.teleai.models.dit.diffsynth_wan_video_vae
import argparse
import hashlib
import importlib
import os
import sys
from pathlib import Path

import torch


def _add_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _parse_pair(value, default):
    if not value:
        return default
    text = value.replace("x", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"invalid pair: {value}")
    return (int(parts[0]), int(parts[1]))


def _build_input(height, width, num_frames, device, dtype):
    total = num_frames * height * width * 3
    values = torch.linspace(-1.0, 1.0, steps=total, dtype=torch.float32, device="cpu")
    images = values.view(1, num_frames, 3, height, width).to(device=device, dtype=dtype)
    return images


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_state_dict(model) -> str | None:
    try:
        sd = model.state_dict()
    except Exception:
        return None
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        h.update(k.encode("utf-8"))
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(str(t.dtype).encode("utf-8"))
            h.update(t.view(torch.uint8).numpy().tobytes())
        else:
            h.update(repr(v).encode("utf-8"))
    return h.hexdigest()


def _sha256_tensor_bytes(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            return obj["model_state"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Minimal DiffSynth VAE init + latent dump.")
    parser.add_argument("--module-root", default="", help="Optional root to add to sys.path.")
    parser.add_argument(
        "--vae-module",
        default="teletron.models.teleai.models.dit.diffsynth_wan_video_vae",
        help="Module path that provides WanVideoVAE.",
    )
    parser.add_argument("--vae-class", default="WanVideoVAE", help="VAE class name.")
    parser.add_argument("--vae-path", required=True, help="VAE weight file path.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--tiled", action="store_true", help="Enable tiled VAE encode.")
    parser.add_argument("--tile-size", default="34,34", help="Tile size, e.g. 34,34")
    parser.add_argument("--tile-stride", default="18,16", help="Tile stride, e.g. 18,16")
    parser.add_argument("--out", default="minimal_vae_latents.pt", help="Output .pt path.")
    args = parser.parse_args()

    if args.module_root:
        _add_sys_path(Path(args.module_root))

    device_str = args.device
    if device_str.startswith("cuda"):
        if device_str == "cuda":
            device_str = "cuda:0"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this demo")
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    module = importlib.import_module(args.vae_module)
    vae_class = getattr(module, args.vae_class, None)
    if vae_class is None:
        raise RuntimeError(f"missing class '{args.vae_class}' in module '{args.vae_module}'")

    vae = vae_class().to(device=device, dtype=torch_dtype).eval().requires_grad_(False)
    vae_path = Path(args.vae_path)
    if not vae_path.exists():
        raise RuntimeError(f"vae weight file not found: {vae_path}")
    raw_state = torch.load(vae_path, map_location="cpu")
    state_dict = _unwrap_state_dict(raw_state)
    target = vae.model if hasattr(vae, "model") else vae
    target.load_state_dict(state_dict, strict=True)

    def _first_param_dtype(m):
        try:
            return next(m.parameters()).dtype
        except Exception:
            return None

    print(f"[minimal] model_param_dtype={_first_param_dtype(target)}")

    tile_size = _parse_pair(args.tile_size, (34, 34))
    tile_stride = _parse_pair(args.tile_stride, (18, 16))
    tiler_kwargs = {"tiled": bool(args.tiled), "tile_size": tile_size, "tile_stride": tile_stride}

    images = _build_input(args.height, args.width, args.num_frames, device, torch_dtype)
    video = images.permute(0, 2, 1, 3, 4)[0]
    print(f"[minimal] input_video_dtype={video.dtype} shape={tuple(video.shape)}")
    print(f"[minimal] video_cthw sha256={_sha256_tensor_bytes(video)}")

    latents = vae.encode([video], device=device, **tiler_kwargs).detach().cpu()
    print(f"[minimal] output_latents_dtype={latents.dtype} shape={tuple(latents.shape)}")

    out_path = Path(args.out)
    if out_path.parent:
        os.makedirs(out_path.parent, exist_ok=True)

    torch.save(
        {
            "latents": latents,
            "input_shape": [1, args.num_frames, 3, args.height, args.width],
            "input_dtype": args.dtype,
            "tiler_kwargs": tiler_kwargs,
            "vae_class": args.vae_class,
            "vae_module": args.vae_module,
            "vae_path": str(vae_path),
            "vae_weight_sha256": _sha256_file(vae_path),
            "vae_state_sha256": _sha256_state_dict(target),
        },
        out_path,
    )
    print(f"[minimal] saved latents to {out_path}")


if __name__ == "__main__":
    main()
