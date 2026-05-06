"""Regression test for ContextParallelMixin's stateless pad/split/gather.

Run with: torchrun --standalone --nproc_per_node N test_context_parallel_mixin.py

Covers:
  1. Bit-identical behavior for the typical (production-like) case where every
     split_input call uses lengths that happen to match — the historical
     latent bug never triggered here, so refactor must be a no-op numerically.
  2. The exact Wan multi-split pattern with mismatched lengths
     (split x dim=1, split freqs dim=0 with shape[0] != x.shape[1]) that
     previously truncated x.  Refactor must produce reference output.
  3. forward_attn-style nested pad: outer split + inner pad-on-different-dim
     + outer gather. Asserts inner pad does not poison outer state.
"""
import os, sys
sys.path.insert(0, "/workspace/teleai_data_tools")
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl", init_method="env://")
rank = dist.get_rank()
world = dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
device = torch.cuda.current_device()

from megatron.core import mpu
mpu.get_context_parallel_world_size = lambda: world
mpu.get_context_parallel_group = lambda: dist.group.WORLD
mpu.get_tensor_model_parallel_world_size = lambda: 1

import teleboost.utils as tu
class _FakeArgs:
    num_attention_heads = 8
tu.get_args = lambda: _FakeArgs()

from teleboost.core.context_parallel.context_parallel_mixin import ContextParallelMixin


class _M(ContextParallelMixin): pass


def _ref_op(x): return torch.tanh(x * 2.0 + 1.0) * 3.0


def _check(name, got, expected):
    """Both must be on the same rank's view of the same logical tensor."""
    same_shape = got.shape == expected.shape
    bitwise = same_shape and torch.equal(got, expected)
    if rank == 0:
        status = "PASS" if bitwise else "FAIL"
        print(f"  [{status}] {name:60s}  got={tuple(got.shape)} expect={tuple(expected.shape)}")
        if not bitwise and same_shape:
            print(f"          max_abs_diff={(got-expected).abs().max().item():.3e}")
    return bitwise


pass_all = True


# ============================================================
# Scenario 1: typical case (lengths match) — must be bit-identical to no-CP
# ============================================================
if rank == 0:
    print(f"\n[CP={world}] Scenario 1: split(x, T=99) → split(freqs, T=99) → gather(x)")

torch.manual_seed(42)
x = torch.randn(1, 99, 32, dtype=torch.bfloat16, device=device); dist.broadcast(x, 0)
freqs = torch.randn(99, 16, dtype=torch.bfloat16, device=device); dist.broadcast(freqs, 0)

m = _M()
y_ref = _ref_op(x)
x_s, x_orig = m.split_input(x, dim=1)
freqs_s, _ = m.split_input(freqs, dim=0)
y_s = _ref_op(x_s)
y_cp = m.gather_output(y_s, dim=1, origin_length=x_orig)
pass_all &= _check("typical: gather(x) matches no-CP reference", y_cp, y_ref)


# ============================================================
# Scenario 2: Wan multi-split pattern with MISMATCHED lengths
# (the bug case — pre-refactor, x's gather narrowed to freqs's length)
# ============================================================
if rank == 0:
    print(f"\n[CP={world}] Scenario 2: split(x, T=99) → split(freqs, T=87) → gather(x)")

torch.manual_seed(123)
x = torch.randn(1, 99, 32, dtype=torch.bfloat16, device=device); dist.broadcast(x, 0)
freqs = torch.randn(87, 16, dtype=torch.bfloat16, device=device); dist.broadcast(freqs, 0)

m2 = _M()
y_ref = _ref_op(x)
x_s, x_orig = m2.split_input(x, dim=1)
_freqs_s, _ = m2.split_input(freqs, dim=0)   # different length, should NOT poison x's gather
y_s = _ref_op(x_s)
y_cp = m2.gather_output(y_s, dim=1, origin_length=x_orig)
pass_all &= _check("multi-split mismatched lengths: x preserved",  y_cp, y_ref)


# ============================================================
# Scenario 3: forward_attn-style nested pad — outer split on dim=1 then
# inner pad on dim=2 (mimicking the SeqAllToAll pad inside forward_attn);
# outer gather must still narrow back to outer x's origin, NOT the inner dim=2 size
# ============================================================
if rank == 0:
    print(f"\n[CP={world}] Scenario 3: split(x, T=99 dim=1) → inner pad(y, dim=2 T=87) → gather(x, dim=1)")

torch.manual_seed(7)
x = torch.randn(1, 99, 32, dtype=torch.bfloat16, device=device); dist.broadcast(x, 0)
y_inner = torch.randn(1, 4, 87, 8, dtype=torch.bfloat16, device=device); dist.broadcast(y_inner, 0)

m3 = _M()
y_ref = _ref_op(x)
x_s, x_orig = m3.split_input(x, dim=1)
y_padded, _y_orig = m3.pad_for_context_parallel(y_inner, dim=2)  # inner pad: returns tuple, doesn't poison
y_s = _ref_op(x_s)
y_cp = m3.gather_output(y_s, dim=1, origin_length=x_orig)
pass_all &= _check("inner pad does not poison outer gather", y_cp, y_ref)


# ============================================================
# Scenario 4: divisible length (no pad needed) — gather should be no-op narrow
# ============================================================
if rank == 0:
    print(f"\n[CP={world}] Scenario 4: split(x, T={world*5}) → gather(x) — divisible, no pad")

T_div = world * 5
torch.manual_seed(11)
x = torch.randn(1, T_div, 32, dtype=torch.bfloat16, device=device); dist.broadcast(x, 0)

m4 = _M()
y_ref = _ref_op(x)
x_s, x_orig = m4.split_input(x, dim=1)
assert x_orig == T_div, f"divisible case should still return origin_length; got {x_orig}"
y_s = _ref_op(x_s)
y_cp = m4.gather_output(y_s, dim=1, origin_length=x_orig)
pass_all &= _check("divisible (no pad): bit-identical reference", y_cp, y_ref)


# ============================================================
# Scenario 5: two independent _M instances — no cross-instance contamination
# (would have been a bug under the class-attribute design)
# ============================================================
if rank == 0:
    print(f"\n[CP={world}] Scenario 5: two mixin instances with different lengths simultaneously")

torch.manual_seed(31)
xa = torch.randn(1, 99, 32, dtype=torch.bfloat16, device=device); dist.broadcast(xa, 0)
xb = torch.randn(1, 71, 32, dtype=torch.bfloat16, device=device); dist.broadcast(xb, 0)

ma = _M(); mb = _M()
ya_ref = _ref_op(xa); yb_ref = _ref_op(xb)

xa_s, ao = ma.split_input(xa, dim=1)
xb_s, bo = mb.split_input(xb, dim=1)
ya_s = _ref_op(xa_s)
yb_s = _ref_op(xb_s)
# interleaved gather — second uses its own origin
ya_cp = ma.gather_output(ya_s, dim=1, origin_length=ao)
yb_cp = mb.gather_output(yb_s, dim=1, origin_length=bo)
pass_all &= _check("instance A preserved through B's split/gather", ya_cp, ya_ref)
pass_all &= _check("instance B preserved through A's split/gather", yb_cp, yb_ref)


dist.barrier()
if rank == 0:
    print()
    print("=" * 60)
    print("OVERALL:", "PASS" if pass_all else "FAIL")
    print("=" * 60)
dist.destroy_process_group()

if not pass_all:
    sys.exit(1)
