"""Auto-apply TeleBoost patches whenever any recipe.teleboost module is imported.

This is critical for Ray worker processes: the main script imports `teleboost`
explicitly, but spawned actors are independent Python processes. They import
modules from `recipe.teleboost.*` (the actor class lives there), which then
loads this `__init__.py`, which loads `teleboost`, which applies the patches
into that worker's verl namespace.
"""
import teleboost  # noqa: F401
