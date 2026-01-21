#!/usr/bin/env python
# Usage:
#   python compare_vae_latents_teletron.py --diffsynth-dump /tmp/diffsynth_vae_latents.pt \
#     --diffsynth-root /path/to/DiffSynth/Megatron_VAST --device cuda
import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import hashlib

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
    """
    Best-effort: hash the *raw bytes* of tensors in state_dict (dtype-agnostic).
    Works for bfloat16 (numpy doesn't support bf16 directly).
    """
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
            # include metadata for safety
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(str(t.dtype).encode("utf-8"))

            # hash raw bytes (dtype-agnostic, supports bf16)
            raw = t.view(torch.uint8).numpy().tobytes()
            h.update(raw)
        else:
            h.update(repr(v).encode("utf-8"))

    return h.hexdigest()



def _resolve_weight_path(p: str, teletron_root: Path, config_path: Path | None = None) -> Path:
    """
    Teletron config里的权重路径可能是相对路径，这里尽量 resolve 成绝对路径。
    优先级：
      1) 绝对路径: 原样
      2) teletron_root / p
      3) config_path.parent / p (如果提供)
    """
    path = Path(p)
    if path.is_absolute():
        return path

    cand1 = (teletron_root / path).resolve()
    if cand1.exists():
        return cand1

    if config_path is not None:
        cand2 = (config_path.parent / path).resolve()
        if cand2.exists():
            return cand2

    # fallback: still return teletron_root-based
    return cand1


def main():
    parser = argparse.ArgumentParser(description="Compare Teletron VAE latents to DiffSynth dump.")
    parser.add_argument(
        "--teletron-root",
        default=os.environ.get("TELETRON_ROOT", ""),
        help="Teletron repo root (defaults to the script directory).",
    )
    parser.add_argument(
        "--diffsynth-root",
        default=os.environ.get("DIFFSYNTH_ROOT", ""),
        help="DiffSynth repo root (set if using DiffSynth VAE init).",
    )
    parser.add_argument(
        "--teletron-config",
        default="",
        help="Path to Teletron config .py (must define `config`).",
    )
    parser.add_argument("--diffsynth-dump", required=True, help="DiffSynth .pt dump file.")
    parser.add_argument(
        "--diffsynth-vae-path",
        default="",
        help="DiffSynth VAE weight file path (defaults to dump payload).",
    )
    parser.add_argument(
        "--diffsynth-vae-model-name",
        default="",
        help="ModelManager model name (defaults to dump payload).",
    )
    parser.add_argument(
        "--use-teletron-encoder",
        action="store_true",
        help="Use Teletron encoder instead of DiffSynth VAE init.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)

    # optional manual override
    parser.add_argument(
        "--teletron-vae-path",
        default="",
        help="Optional override: Teletron-side VAE weight file path to hash-compare with DiffSynth dump.",
    )
    parser.add_argument(
        "--strict-weight-hash",
        action="store_true",
        help="If set, mismatch in vae_weight_sha256 will raise an error.",
    )
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

    diffsynth_vae_weight_sha256 = payload.get("vae_weight_sha256", None)
    diffsynth_vae_state_sha256 = payload.get("vae_state_sha256", None)
    diffsynth_vae_path = payload.get("vae_path", None)
    diffsynth_vae_model_name = payload.get("vae_model_name", None)

    if not torch.is_tensor(latents_ref):
        raise RuntimeError("diffsynth dump missing latents tensor")
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 5:
        raise RuntimeError("diffsynth dump missing input_shape")
    _, num_frames, _, height, width = input_shape

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[input_dtype]
    images = _build_input(height, width, num_frames, device, torch_dtype)

    dump_tiler = payload.get("tiler_kwargs", {})
    if not isinstance(dump_tiler, dict):
        dump_tiler = {}

    if args.use_teletron_encoder:
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
        teletron_batch = {"images": images}
        tele_latents = teletron_encoder.work_fn["latents"](batch=teletron_batch).detach().cpu()
        mode_label = "teletron"
    else:
        diffsynth_root = Path(args.diffsynth_root) if args.diffsynth_root else None
        if diffsynth_root is None:
            raise RuntimeError("diffsynth root not found; set --diffsynth-root or DIFFSYNTH_ROOT")
        _add_sys_path(diffsynth_root)
        vae_path = args.diffsynth_vae_path or diffsynth_vae_path
        if not vae_path:
            raise RuntimeError("diffsynth vae path missing; set --diffsynth-vae-path")
        vae_model_name = args.diffsynth_vae_model_name or diffsynth_vae_model_name or "wan_video_vae"
        from diffsynth.models.model_manager import ModelManager

        manager = ModelManager(torch_dtype=torch_dtype, device=device, file_path_list=[vae_path])
        vae = manager.fetch_model(vae_model_name)
        if vae is None:
            available = ", ".join(manager.model_name)
            raise RuntimeError(f"diffsynth vae model '{vae_model_name}' not found. available: {available}")
        vae_weight_sha256 = _sha256_file(Path(vae_path))
        vae_state_sha256 = _sha256_state_dict(getattr(vae, "model", vae))
        if diffsynth_vae_weight_sha256 is not None:
            ok = (vae_weight_sha256 == diffsynth_vae_weight_sha256)
            print(f"[diffsynth] weight_hash_match={ok}")
            if (not ok) and args.strict_weight_hash:
                raise RuntimeError("VAE weight SHA256 mismatch (file-level).")
        if diffsynth_vae_state_sha256 is not None:
            print(f"[diffsynth] state_hash_match={vae_state_sha256 == diffsynth_vae_state_sha256}")
        video = images.permute(0, 2, 1, 3, 4)[0]
        tele_latents = vae.encode([video], device=device, **dump_tiler).detach().cpu()
        mode_label = "diffsynth"

    result = _compare_tensors(latents_ref, tele_latents, args.rtol, args.atol)
    print(f"[{mode_label}] latents shape={tuple(tele_latents.shape)} dtype={tele_latents.dtype}")
    print(f"[diffsynth] latents shape={tuple(latents_ref.shape)} dtype={latents_ref.dtype}")
    print(f"[compare] allclose={result['allclose']} max_abs={result['max_abs']} mean_abs={result['mean_abs']}")



if __name__ == "__main__":
    main()
