import re
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


def _tensor_bytes_sha256(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def _to_cpu_detached(x: Any) -> Any:
    """Detach tensors and move to CPU; keep structure for (list/tuple/dict)."""
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu_detached(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_cpu_detached(v) for k, v in x.items()}
    return x


def _flatten_tensors(x: Any, prefix: str = "") -> List[Tuple[str, torch.Tensor]]:
    """
    Flatten nested structure to list of (key, tensor).
    key is a stable path like "out", "out.0", "out.key".
    """
    out = []
    if torch.is_tensor(x):
        out.append((prefix or "tensor", x))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.extend(_flatten_tensors(v, f"{prefix}.{i}" if prefix else str(i)))
    elif isinstance(x, dict):
        for k in sorted(x.keys(), key=lambda z: str(z)):
            v = x[k]
            out.extend(_flatten_tensors(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


def _sample_tensor(
    t: torch.Tensor,
    mode: str = "none",
    max_elems: int = 200_000,
    seed: int = 0,
) -> torch.Tensor:
    """
    Reduce tensor size for saving.
    mode:
      - "none": save full tensor (may be huge)
      - "head": save first max_elems in flattened order
      - "rand": random sample of max_elems elements (deterministic with seed)
    """
    t = t.detach()
    if mode == "none":
        return t
    flat = t.flatten()
    n = flat.numel()
    if n <= max_elems:
        return t
    if mode == "head":
        return flat[:max_elems].clone()
    if mode == "rand":
        g = torch.Generator(device=flat.device)
        g.manual_seed(seed)
        idx = torch.randperm(n, generator=g, device=flat.device)[:max_elems]
        return flat[idx].clone()
    raise ValueError(f"unknown sample mode: {mode}")


@dataclass
class TraceTensorMeta:
    name: str              # module qualified name
    kind: str              # "in" or "out"
    key: str               # nested key inside in/out structure
    shape: List[int]
    dtype: str
    device: str
    sha256: str
    # stats in fp32 to compare quickly
    min: float
    max: float
    mean: float


class ForwardTraceRecorder:
    """
    Recursively attaches forward hooks to a model and records per-module outputs.
    Designed for deterministic debugging and cross-system comparisons.

    Features:
    - include/exclude name regex
    - skip container-like modules (Sequential, ModuleList, etc.)
    - record inputs and/or outputs
    - sample large tensors (head/rand) to control size
    - store in float32 (optional) for stable comparison
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        name: str,
        record_inputs: bool = False,
        record_outputs: bool = True,
        include_name_regex: Optional[str] = None,
        exclude_name_regex: Optional[str] = None,
        include_module_types: Optional[Tuple[type, ...]] = None,
        exclude_module_types: Optional[Tuple[type, ...]] = (nn.Sequential, nn.ModuleList, nn.ModuleDict),
        sample_mode: str = "none",   # "none" | "head" | "rand"
        sample_max_elems: int = 200_000,
        sample_seed: int = 0,
        cast_float32: bool = False,  # if True, store tensors in fp32 (recommended for compare)
        save_stats_only: bool = False,  # if True, do not store tensor values; only store meta+stats
        max_modules: Optional[int] = None,  # limit number of hooked modules
        verbose: bool = False,
    ):
        self.model = model
        self.name = name
        self.record_inputs = record_inputs
        self.record_outputs = record_outputs
        self.include_re = re.compile(include_name_regex) if include_name_regex else None
        self.exclude_re = re.compile(exclude_name_regex) if exclude_name_regex else None
        self.include_types = include_module_types
        self.exclude_types = exclude_module_types
        self.sample_mode = sample_mode
        self.sample_max_elems = sample_max_elems
        self.sample_seed = sample_seed
        self.cast_float32 = cast_float32
        self.save_stats_only = save_stats_only
        self.max_modules = max_modules
        self.verbose = verbose

        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.records: Dict[str, Dict[str, Any]] = {}  # module_name -> {"inputs":..., "outputs":..., "meta":[...]}
        self._module_count = 0
        self._start_time = None

    def _should_hook(self, module_name: str, module: nn.Module) -> bool:
        if module is self.model:
            return False  # skip root to reduce redundancy; change if you want
        if self.include_re and not self.include_re.search(module_name):
            return False
        if self.exclude_re and self.exclude_re.search(module_name):
            return False
        if self.include_types and not isinstance(module, self.include_types):
            return False
        if self.exclude_types and isinstance(module, self.exclude_types):
            return False
        # Don't hook modules with no parameters AND no buffers? optional; leave as-is for full trace.
        return True

    def _hook_fn(self, module_name: str):
        def fn(module: nn.Module, inputs: Tuple[Any, ...], outputs: Any):
            rec = self.records.setdefault(module_name, {"inputs": None, "outputs": None, "meta": []})

            def process(kind: str, obj: Any):
                cpu_obj = _to_cpu_detached(obj)

                # For stable compare, cast to fp32 on CPU (optional)
                flat = _flatten_tensors(cpu_obj, prefix="")
                processed_store = {}
                for key, t in flat:
                    if not torch.is_tensor(t):
                        continue
                    tt = t.contiguous()
                    if self.cast_float32:
                        tt = tt.float()

                    # sample to reduce size
                    sampled = _sample_tensor(tt, mode=self.sample_mode, max_elems=self.sample_max_elems, seed=self.sample_seed)

                    meta = TraceTensorMeta(
                        name=module_name,
                        kind=kind,
                        key=key,
                        shape=list(tt.shape),
                        dtype=str(tt.dtype),
                        device=str(tt.device),
                        sha256=_tensor_bytes_sha256(sampled),
                        min=float(tt.float().min().item()) if tt.numel() else 0.0,
                        max=float(tt.float().max().item()) if tt.numel() else 0.0,
                        mean=float(tt.float().mean().item()) if tt.numel() else 0.0,
                    )
                    rec["meta"].append(asdict(meta))

                    if not self.save_stats_only:
                        processed_store[key] = sampled  # sampled tensor (maybe full)

                return processed_store

            if self.record_inputs:
                # inputs is a tuple; store flattened form
                rec["inputs"] = process("in", inputs)
            if self.record_outputs:
                rec["outputs"] = process("out", outputs)

        return fn

    def install(self) -> None:
        self._start_time = time.time()
        for module_name, module in self.model.named_modules():
            if self.max_modules is not None and self._module_count >= self.max_modules:
                break
            if not self._should_hook(module_name, module):
                continue
            h = module.register_forward_hook(self._hook_fn(module_name))
            self.handles.append(h)
            self._module_count += 1
            if self.verbose:
                print(f"[trace:{self.name}] hook {module_name}: {module.__class__.__name__}")

        if self.verbose:
            print(f"[trace:{self.name}] installed hooks on {self._module_count} modules")

    def remove(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    def save(self, path: Union[str, Path], extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "trace_name": self.name,
            "num_modules_hooked": self._module_count,
            "elapsed_sec": (time.time() - self._start_time) if self._start_time else None,
            "record_inputs": self.record_inputs,
            "record_outputs": self.record_outputs,
            "cast_float32": self.cast_float32,
            "sample_mode": self.sample_mode,
            "sample_max_elems": self.sample_max_elems,
            "save_stats_only": self.save_stats_only,
            "records": self.records,
            "extra": extra or {},
        }
        torch.save(payload, str(path))
        if self.verbose:
            print(f"[trace:{self.name}] saved to {path}")
