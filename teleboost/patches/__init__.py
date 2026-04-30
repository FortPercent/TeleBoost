"""Runtime patches that overlay TeleBoost-specific fixes onto upstream verl.

Import this package once during process startup (TeleBoost's `teleboost/__init__.py`
does it implicitly) to apply every patch. Patches are idempotent.
"""
from teleboost.patches.debug_extras import apply as _apply_debug_extras
from teleboost.patches.ulysses_cp_fix import apply as _apply_cp_fix
from teleboost.patches.wan_save_compat import apply as _apply_wan_save_compat

_APPLIED = False


def apply() -> None:
    """Apply every TeleBoost patch over upstream verl@v0.4.0. Idempotent.

    Each patch is a small, targeted attribute-level injection into a verl
    namespace - no whole-module replacement, no sys.modules tricks. If you
    need to add new project-specific behaviour, write a new patch module
    that defines `apply()` and append it here.
    """
    global _APPLIED
    if _APPLIED:
        return
    _apply_debug_extras()
    _apply_cp_fix()
    _apply_wan_save_compat()
    _APPLIED = True
