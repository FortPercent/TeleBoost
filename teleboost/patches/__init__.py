"""Runtime patches that overlay TeleBoost-specific fixes onto upstream verl.

Import this package once during process startup (TeleBoost's `teleboost/__init__.py`
does it implicitly) to apply every patch. Patches are idempotent.
"""
from teleboost.patches.ulysses_cp_fix import apply as _apply_cp_fix

_APPLIED = False


def apply() -> None:
    """Apply every TeleBoost patch over upstream verl. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    _apply_cp_fix()
    _APPLIED = True
