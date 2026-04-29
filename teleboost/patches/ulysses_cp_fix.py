"""Fix the modulation grad double-reduce bug under Ulysses sequence parallel.

Upstream `register_cp_grad_reduce_hook` matches every parameter with "blocks"
in its name and does a SUM all-reduce of the grad. For wan-style transformer
blocks, the modulation parameter (shift/scale/gate) has its grad already
SUM-allreduced inside `ModulateWithCPGradReduce.backward` /
`GateWithGradReduce.backward`. Letting the hook fire on it duplicates the
reduce, producing `0.5 * sp_size * G_full` instead of `G_full` once the
historical `mul_(0.5)` post-backward compensation is removed.

Fix: skip the modulation params in the hook. Mathematically equivalent to
"do not double-reduce". Verified bit-exact in fp32 and within the bf16 reduce
floor (~1.5e-4 at sp=8) by `tests/special_distributed/test_cp_grad_reduce.py`.
"""
from __future__ import annotations


def apply() -> None:
    """Inject TeleBoost cp-aware autograd Functions + fixed register hook into
    verl.utils.ulysses. Wan transformer blocks then call them via
    `from verl.utils.ulysses import gate_with_cp_grad_reduce` etc.
    """
    import torch
    import torch.distributed as dist
    import verl.utils.ulysses as _u

    # 1. Inject the cp-aware autograd Function entry points (upstream-missing).
    from teleboost.utils._ulysses_cp import (
        GateWithGradReduce,
        ModulateWithCPGradReduce,
        gate_with_cp_grad_reduce,
        modulate_with_cp_grad_reduce,
    )
    for name, value in [
        ("GateWithGradReduce", GateWithGradReduce),
        ("ModulateWithCPGradReduce", ModulateWithCPGradReduce),
        ("gate_with_cp_grad_reduce", gate_with_cp_grad_reduce),
        ("modulate_with_cp_grad_reduce", modulate_with_cp_grad_reduce),
    ]:
        if not hasattr(_u, name):
            setattr(_u, name, value)

    # 2. Replace register_cp_grad_reduce_hook to skip modulation params (root-cause
    #    fix for the cp grad double-reduce bug; see fix b0cecf7a / b311fbfc).
    def register_cp_grad_reduce_hook(model):
        def _cp_grad_reduce(grad):
            with torch.no_grad():
                dist.all_reduce(
                    grad,
                    op=dist.ReduceOp.SUM,
                    group=_u.get_ulysses_sequence_parallel_group(),
                )
                return grad

        for name, param in model.named_parameters():
            # modulation params are already SUM-allreduced inside Modulate/Gate
            # WithCPGradReduce.backward; skipping here avoids double-reduce.
            if "blocks" in name and "modulation" not in name.lower():
                param.register_hook(_cp_grad_reduce)

    _u.register_cp_grad_reduce_hook = register_cp_grad_reduce_hook
