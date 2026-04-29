#!/usr/bin/env python
import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import hashlib

import torch

# 你已保存的 hook 工具类
# forward_trace.py 里需要暴露 ForwardTraceRecorder
from my_utils import ForwardTraceRecorder


def _add_sys_path(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


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
    encoder_cfg = config.get("model_config", {}).get("encoder", {})
    encoder_name = encoder_cfg.get("type")
    if not encoder_name:
        raise RuntimeError("teletron encoder config missing type field")
    enc = get_encoder(name=encoder_name, device=device)
    enc.setup()
    return enc


def _build_input(height, width, num_frames, device, dtype):
    total = num_frames * height * width * 3
    values = torch.linspace(-1.0, 1.0, steps=total, dtype=torch.float32, device="cpu")
    images = values.view(1, num_frames, 3, height, width).to(device=device, dtype=dtype)
    return images


def _sha256_tensor_bytes(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def _first_param_dtype(m) -> str | None:
    try:
        return str(next(m.parameters()).dtype)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Dump Teletron VAE forward traces (per-module outputs via hooks).")

    # 默认不需要传 root/config：脚本放在 teletron repo 里即可
    parser.add_argument("--teletron-root", default="", help="(Optional) Teletron repo root. Default: script's parent dir.")
    parser.add_argument("--teletron-config", default="", help="(Optional) Teletron config .py. Default: examples/teleai/config/wan_dpo.py")

    parser.add_argument("--diffsynth-dump", required=True, help="DiffSynth .pt dump (for input_shape/input_dtype/tiler_kwargs)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # trace settings
    parser.add_argument("--out", default="teletron_vae_trace.pt", help="Output trace .pt path.")
    parser.add_argument("--record-inputs", action="store_true", help="Record inputs too (can be huge).")
    parser.add_argument("--include", default="", help="Regex: only hook module names matching this.")
    parser.add_argument("--exclude", default="", help="Regex: exclude module names matching this.")
    parser.add_argument("--sample-mode", default="head", choices=["none", "head", "rand"])
    parser.add_argument("--max-elems", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)

    # 关键：默认不 cast，保留真实 bf16 输出；如需存 fp32 才开这个
    parser.add_argument("--trace-store-fp32", action="store_true", help="Store tensor values in fp32 (NOT recommended for exact byte-compare).")
    parser.add_argument("--stats-only", action="store_true", help="Only save meta/stats, do not save tensor values.")
    parser.add_argument("--max-modules", type=int, default=0, help="Limit hooked modules (0=unlimited).")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # ---- resolve teletron root/config defaults ----
    if args.teletron_root:
        teletron_root = Path(args.teletron_root)
    else:
        # 脚本在 teletron repo 内，例如 teletron/tools/teletron_vae_trace.py
        teletron_root = Path(__file__).resolve().parents[2]
    _add_sys_path(teletron_root)

    if args.teletron_config:
        config_path = Path(args.teletron_config)
        if not config_path.is_absolute():
            config_path = (teletron_root / config_path).resolve()
    else:
        config_path = (teletron_root / "examples" / "teleai" / "config" / "wan_dpo.py").resolve()

    if not config_path.exists():
        raise RuntimeError(f"Teletron config not found: {config_path}")

    # ---- device ----
    device_str = args.device
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if device_str == "cuda":
            device_str = "cuda:0"
        device = torch.device(device_str)
        torch.cuda.set_device(device.index if device.index is not None else 0)
    else:
        device = torch.device(device_str)

    # ---- load teletron config & build encoder ----
    config = _load_config_from_py(config_path)
    _patch_set_config(config)
    teletron_encoder = _build_teletron_encoder(config, device, args.encoder_dtype)

    if teletron_encoder.vae is None:
        raise RuntimeError("teletron_encoder.vae is None")

    tele_vae = teletron_encoder.vae
    tele_vae_model = getattr(tele_vae, "model", tele_vae)

    # ---- load diffsynth dump for input spec ----
    payload = torch.load(args.diffsynth_dump, map_location="cpu")
    input_shape = payload.get("input_shape")
    input_dtype = payload.get("input_dtype", args.encoder_dtype)
    tiler_kwargs = payload.get("tiler_kwargs", {})

    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 5:
        raise RuntimeError("diffsynth dump missing input_shape")
    _, num_frames, _, height, width = input_shape

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[input_dtype]
    images = _build_input(height, width, num_frames, device, torch_dtype)

    # ---- input hash sanity ----
    video_cthw = images.permute(0, 2, 1, 3, 4)[0]    # (C,T,H,W)
    input_bcthw = images.permute(0, 2, 1, 3, 4)      # (B,C,T,H,W)
    print(f"[teletron] teletron_root={teletron_root}")
    print(f"[teletron] config={config_path}")
    print(f"[teletron] tiler_kwargs(from dump)={tiler_kwargs}")
    print(f"[teletron] images dtype={images.dtype} shape={tuple(images.shape)}")
    print(f"[teletron] video_cthw sha256={_sha256_tensor_bytes(video_cthw)}")
    print(f"[teletron] input_bcthw sha256={_sha256_tensor_bytes(input_bcthw)}")
    print(f"[teletron] vae_param_dtype={_first_param_dtype(tele_vae_model)}")

    # ---- install hooks ----
    max_modules = None if args.max_modules == 0 else int(args.max_modules)

    recorder = ForwardTraceRecorder(
        tele_vae_model,
        name="teletron.vae_model",
        record_inputs=bool(args.record_inputs),
        record_outputs=True,
        include_name_regex=args.include or None,
        exclude_name_regex=args.exclude or None,
        sample_mode=args.sample_mode,
        sample_max_elems=int(args.max_elems),
        sample_seed=int(args.seed),
        # 这里用 “是否存 fp32” 作为开关；默认 False 保留 bf16 原始输出
        cast_float32=bool(args.trace_store_fp32),
        save_stats_only=bool(args.stats_only),
        max_modules=max_modules,
        verbose=bool(args.verbose),
    )
    recorder.install()

    # ---- trigger encode (this is what you want to compare later) ----
    with torch.no_grad():
        inp = images.permute(0, 2, 1, 3, 4).to(device=torch.cuda.current_device(), dtype=images.dtype)  # (B,C,T,H,W)
        _ = tele_vae.encode(inp, device=torch.cuda.current_device(), **tiler_kwargs)

    recorder.remove()

    # ---- save ----
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extra = {
        "teletron_root": str(teletron_root),
        "teletron_config": str(config_path),
        "device": str(device),
        "input_shape": list(input_shape),
        "input_dtype": input_dtype,
        "tiler_kwargs": tiler_kwargs,
        "video_cthw_sha256": _sha256_tensor_bytes(video_cthw),
        "input_bcthw_sha256": _sha256_tensor_bytes(input_bcthw),
        "vae_param_dtype": _first_param_dtype(tele_vae_model),
        "trace_store_fp32": bool(args.trace_store_fp32),
        "sample_mode": args.sample_mode,
        "max_elems": int(args.max_elems),
    }
    recorder.save(out_path, extra=extra)
    print(f"[teletron] saved trace to {out_path}")


if __name__ == "__main__":
    main()
