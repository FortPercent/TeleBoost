# Public surface for OSS users.
#
# `DPODatasetBase` is self-contained (only depends on torch) so we eager-import
# it; the others are lazy because of pre-existing circular imports —
# teleboost.utils.config imports teleboost.datasets.utils, which imports the
# datasets package; eagerly importing FakeDataset / DATASETS here re-enters
# teleboost.utils before its __init__ finishes.
from .dpo_base import DPODatasetBase

__all__ = ["DPODatasetBase", "FakeDataset", "FakeDPODataset",
           "DATASETS", "build_dataset"]


def __getattr__(name):
    """PEP 562 lazy attribute access — resolves once teleboost.utils is ready."""
    if name == "FakeDataset":
        from .fake_dataset import FakeDataset
        return FakeDataset
    if name == "FakeDPODataset":
        from .fake_dataset import FakeDPODataset
        return FakeDPODataset
    if name in ("DATASETS", "build_dataset"):
        from .build import DATASETS, build_dataset
        return {"DATASETS": DATASETS, "build_dataset": build_dataset}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
