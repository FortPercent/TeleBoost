import os

import torch


class DumpTensorIO:
    def __init__(
        self,
        tensor_dir_env="WAN_DPO_PREVAE_TENSOR_DIR",
        default_tensor_dir="dpo_dumps",
    ):
        self.tensor_dir = os.environ.get(tensor_dir_env, default_tensor_dir)
        self._cache = {}

    def _safe_tag(self, tag):
        return str(tag).replace("/", "_").replace(" ", "_")

    def _tensor_path(self, dump_id, tag, rank):
        safe_tag = self._safe_tag(tag)
        return os.path.join(self.tensor_dir, f"{int(dump_id):04d}_{safe_tag}_rank{int(rank)}.pt")

    def load_tensors(self, dump_id, tag, rank, map_location="cpu"):
        path = self._tensor_path(dump_id, tag, rank)
        if path in self._cache:
            return self._cache[path]
        if not os.path.exists(path):
            return None
        payload = torch.load(path, map_location=map_location)
        self._cache[path] = payload
        return payload

    def compare_tensors(self, expected, actual, rtol=1e-5, atol=1e-8):
        if expected is None:
            return {"missing": True}
        if not torch.is_tensor(expected) or not torch.is_tensor(actual):
            return {"type_mismatch": True}
        if expected.shape != actual.shape:
            return {
                "shape_mismatch": True,
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
            }
        diff = (actual - expected).abs()
        return {
            "missing": False,
            "allclose": bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)),
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
        }
