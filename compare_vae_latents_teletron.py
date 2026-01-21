#!/usr/bin/env python
# Usage:
#   python compare_vae_latents_teletron.py --diffsynth-dump /tmp/diffsynth_vae_latents.pt --device cuda
import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def _add_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_config_from_py(path: Path):
    spec = importlib.util.spec_from_file_location("teletron_config_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "config", None)
    if not isinstance(config, dict):
        raise RuntimeError(f"config not found or not a dict in {path}")
    from teletron.utils.config import load_config

    return load_config(config)


def _patch_set_config(config):
    import teletron.utils.config as teletron_config
    import teletron.utils as teletron_utils

    def _set_config():
        return config

    teletron_config.set_config = _set_config
    teletron_utils.set_config = _set_config
    try:
        import teletron.models.teleai.teleai_encoder_utils as teleai_encoder_utils

        teleai_encoder_utils.set_config = _set_config
    except Exception:
        pass
    try:
        import teletron.models.teleai.teleai_encoder as teleai_encoder

        teleai_encoder.set_config = _set_config
    except Exception:
        pass


def _build_teletron_encoder(config, device, encoder_dtype):
    from teletron.utils.global_vars import set_global_args
    from teletron.models.encoder_registry import get_encoder

    args = SimpleNamespace(
        encoder_dtype=encoder_dtype,
        consumer_models_num=1,
        micro_batch_size=1,
        negative_prompt="",
    )
    set_global_args(args)
    encoder_config = config.get("model_config", {}).get("encoder", {})
    encoder_name = encoder_config.get("type")
    if not encoder_name:
        raise RuntimeError("teletron encoder config missing type field")
    encoder = get_encoder(name=encoder_name, device=device)
    encoder.setup()
    return encoder, encoder_config


def _build_input(height, width, num_frames, device, dtype):
    total = num_frames * height * width * 3
    values = torch.linspace(-1.0, 1.0, steps=total, dtype=torch.float32, device="cpu")
    images = values.view(1, num_frames, 3, height, width).to(device=device, dtype=dtype)
    return images


def _compare_tensors(a, b, rtol, atol):
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        raise RuntimeError("compare requires tensors")
    if a.shape != b.shape:
        raise RuntimeError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (a.float() - b.float()).abs()
    return {
        "allclose": bool(torch.allclose(a, b, rtol=rtol, atol=atol)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Teletron VAE latents to DiffSynth dump.")
    parser.add_argument(
        "--teletron-root",
        default=os.environ.get("TELETRON_ROOT", ""),
        help="Teletron repo root (defaults to the script directory).",
    )
    parser.add_argument(
        "--teletron-config",
        default="",
        help="Path to Teletron config .py (must define `config`).",
    )
    parser.add_argument("--diffsynth-dump", required=True, help="DiffSynth .pt dump file.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    device_str = args.device
    if device_str.startswith("cuda"):
        if device_str == "cuda":
            device_str = "cuda:0"
        device = torch.device(device_str)
        torch.cuda.set_device(device.index if device.index is not None else 0)
    else:
        device = torch.device(device_str)

    # Load DiffSynth dump
    payload = torch.load(args.diffsynth_dump, map_location="cpu")
    latents_ref = payload.get("latents")
    input_shape = payload.get("input_shape")
    input_dtype = payload.get("input_dtype", args.encoder_dtype)

    if not torch.is_tensor(latents_ref):
        raise RuntimeError("diffsynth dump missing latents tensor")
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 5:
        raise RuntimeError("diffsynth dump missing input_shape")
    _, num_frames, _, height, width = input_shape

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[input_dtype]
    images = _build_input(height, width, num_frames, device, torch_dtype)

    teletron_root = Path(args.teletron_root) if args.teletron_root else Path(__file__).resolve().parent
    _add_sys_path(teletron_root)
    if not args.teletron_config:
        args.teletron_config = str(teletron_root / "examples" / "teleai" / "config" / "wan_dpo.py")
    config_path = Path(args.teletron_config)
    if not config_path.exists():
        raise RuntimeError(f"Teletron config not found: {config_path}")
    config = _load_config_from_py(config_path)
    _patch_set_config(config)
    teletron_encoder, _ = _build_teletron_encoder(config, device, args.encoder_dtype)
    dump_tiler = payload.get("tiler_kwargs", {})
    if not isinstance(dump_tiler, dict):
        dump_tiler = {}
    dump_tiler.setdefault("tiled", False)
    dump_tiler.setdefault("tile_size", (34, 34))
    dump_tiler.setdefault("tile_stride", (18, 16))
    video = images.permute(0, 2, 1, 3, 4)[0]
    tele_latents = teletron_encoder.vae.encode([video], device=device, **dump_tiler).detach().cpu()
    mode_label = "teletron"

    result = _compare_tensors(latents_ref, tele_latents, args.rtol, args.atol)
    print(f"[{mode_label}] latents shape={tuple(tele_latents.shape)} dtype={tele_latents.dtype}")
    print(f"[diffsynth] latents shape={tuple(latents_ref.shape)} dtype={latents_ref.dtype}")
    print(f"[compare] allclose={result['allclose']} max_abs={result['max_abs']} mean_abs={result['mean_abs']}")



if __name__ == "__main__":
    main()
