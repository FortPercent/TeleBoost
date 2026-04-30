"""Runtime patches that overlay TeleBoost-specific fixes onto upstream verl.

Import this package once during process startup (TeleBoost's `teleboost/__init__.py`
does it implicitly) to apply every patch. Patches are idempotent.
"""
from teleboost.patches.debug_extras import apply as _apply_debug_extras
from teleboost.patches.module_overrides import apply as _apply_module_overrides
from teleboost.patches.ulysses_cp_fix import apply as _apply_cp_fix

_APPLIED = False


def apply() -> None:
    """Apply every TeleBoost patch over upstream verl. Idempotent.

    Order matters: cp_fix injects gate_with_cp_grad_reduce et al into
    verl.utils.ulysses; module_overrides triggers importing teleboost
    modules (e.g. teleboost.models.transformers.wan) whose top-level
    `from verl.utils.ulysses import gate_with_cp_grad_reduce` requires
    the cp_fix injections to have happened first.
    """
    global _APPLIED
    if _APPLIED:
        return
    _apply_debug_extras()
    _apply_cp_fix()
    _apply_module_overrides()
    _APPLIED = True
