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
    """Replace verl.utils.ulysses.register_cp_grad_reduce_hook in-place."""
    import torch
    import torch.distributed as dist
    import verl.utils.ulysses as _u

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
            # modulation params are already SUM-allreduced inside
            # ModulateWithCPGradReduce / GateWithGradReduce backward; skipping
            # here avoids double-reduce that would scale grad by sp_size.
            if "blocks" in name and "modulation" not in name.lower():
                param.register_hook(_cp_grad_reduce)

    _u.register_cp_grad_reduce_hook = register_cp_grad_reduce_hook
