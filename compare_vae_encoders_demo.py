#!/usr/bin/env python
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


def _build_diffsynth_vae(vae_path, model_name, device, torch_dtype):
    from diffsynth.models.model_manager import ModelManager

    manager = ModelManager(torch_dtype=torch_dtype, device=device, file_path_list=[vae_path])
    vae = manager.fetch_model(model_name)
    if vae is None:
        available = ", ".join(manager.model_name)
        raise RuntimeError(f"diffsynth vae model '{model_name}' not found. available: {available}")
    return vae


def _compare_tensors(name, a, b, rtol, atol):
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        raise RuntimeError(f"{name} compare requires tensors")
    if a.shape != b.shape:
        raise RuntimeError(f"{name} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (a.float() - b.float()).abs()
    print(
        f"[compare] {name} allclose={torch.allclose(a, b, rtol=rtol, atol=atol)} "
        f"max_abs={float(diff.max())} mean_abs={float(diff.mean())}"
    )


def main():
    root = Path(__file__).resolve().parent
    teletron_root = root / "Teletron-clean" / "Teletron"
    diffsynth_root = root / "DiffSynth" / "Megatron_VAST"
    _add_sys_path(teletron_root)
    _add_sys_path(diffsynth_root)

    parser = argparse.ArgumentParser(description="Minimal Teletron vs DiffSynth VAE compare demo.")
    parser.add_argument(
        "--teletron-config",
        default=str(teletron_root / "examples" / "teleai" / "config" / "wan_dpo.py"),
        help="Path to Teletron config .py (must define `config`).",
    )
    parser.add_argument(
        "--diffsynth-vae-path",
        default="",
        help="DiffSynth VAE weight file path. Defaults to Teletron config encoder.vae.path.",
    )
    parser.add_argument(
        "--diffsynth-vae-model-name",
        default="wan_video_vae",
        help="ModelManager model name for DiffSynth VAE.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    config_path = Path(args.teletron_config)
    if not config_path.exists():
        raise RuntimeError(f"Teletron config not found: {config_path}")
    config = _load_config_from_py(config_path)
    _patch_set_config(config)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    encoder_dtype = args.encoder_dtype
    teletron_encoder, encoder_config = _build_teletron_encoder(config, device, encoder_dtype)
    vae_cfg = encoder_config.get("vae", {})
    vae_path = args.diffsynth_vae_path or (vae_cfg.get("path") if isinstance(vae_cfg, dict) else "")
    if not vae_path:
        raise RuntimeError("DiffSynth VAE path missing; set --diffsynth-vae-path")

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[encoder_dtype]
    diffsynth_vae = _build_diffsynth_vae(vae_path, args.diffsynth_vae_model_name, device, torch_dtype)

    torch.manual_seed(0)
    total = args.num_frames * args.height * args.width * 3
    values = torch.linspace(-1.0, 1.0, steps=total, device=device, dtype=torch_dtype)
    images = values.view(1, args.num_frames, 3, args.height, args.width)

    teletron_batch = {"images": images}
    teletron_latents = teletron_encoder.work_fn["latents"](batch=teletron_batch)
    video = images.permute(0, 2, 1, 3, 4)[0]
    tiler_kwargs = vae_cfg.get("tiler_kwargs", {}) if isinstance(vae_cfg, dict) else {}
    if tiler_kwargs is None:
        tiler_kwargs = dict(
            tiled=False,
            tile_size=(34, 34),
            tile_stride=(18, 16),
        )
    diffsynth_latents = diffsynth_vae.encode([video], device=device, **tiler_kwargs)

    print(f"[teletron] latents shape={tuple(teletron_latents.shape)} dtype={teletron_latents.dtype}")
    print(f"[diffsynth] latents shape={tuple(diffsynth_latents.shape)} dtype={diffsynth_latents.dtype}")
    _compare_tensors("vae_latents", teletron_latents, diffsynth_latents, args.rtol, args.atol)


if __name__ == "__main__":
    main()
