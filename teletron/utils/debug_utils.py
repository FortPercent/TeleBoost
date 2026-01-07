import json
import os
from datetime import datetime

import numpy as np
import torch


def summarize_object(obj, max_items=10, max_depth=3, max_repr=200):
    """Return a JSON-serializable summary of an object's structure."""
    if max_depth <= 0:
        return {"type": type(obj).__name__}

    if torch.is_tensor(obj):
        return {
            "type": "torch.Tensor",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "device": str(obj.device),
            "requires_grad": bool(obj.requires_grad),
        }

    if isinstance(obj, np.ndarray):
        return {
            "type": "np.ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }

    if isinstance(obj, np.generic):
        return {"type": type(obj).__name__, "value": obj.item()}

    if isinstance(obj, dict):
        items = list(obj.items())
        summarized = {
            str(k): summarize_object(v, max_items=max_items, max_depth=max_depth - 1, max_repr=max_repr)
            for k, v in items[:max_items]
        }
        payload = {"type": "dict", "len": len(items), "items": summarized}
        if len(items) > max_items:
            payload["truncated"] = len(items) - max_items
        return payload

    if isinstance(obj, (list, tuple, set)):
        items = list(obj)
        summarized = [
            summarize_object(v, max_items=max_items, max_depth=max_depth - 1, max_repr=max_repr)
            for v in items[:max_items]
        ]
        payload = {"type": type(obj).__name__, "len": len(items), "items": summarized}
        if len(items) > max_items:
            payload["truncated"] = len(items) - max_items
        return payload

    if isinstance(obj, bytes):
        return {"type": "bytes", "len": len(obj), "repr": repr(obj[:max_repr])}

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return {"type": type(obj).__name__, "value": obj}

    return {"type": type(obj).__name__, "repr": repr(obj)[:max_repr]}


def dump_object_summary(
    obj,
    output_path,
    meta=None,
    max_items=10,
    max_depth=3,
    max_repr=200,
    ensure_ascii=True,
    logger=None,
):
    """Append a summarized object dump to a JSONL file."""
    try:
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "summary": summarize_object(
                obj, max_items=max_items, max_depth=max_depth, max_repr=max_repr
            ),
        }
        if meta:
            payload.update(meta)

        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=ensure_ascii) + "\n")
        return True
    except Exception as exc:
        if logger is not None:
            logger.debug(f"Object summary dump failed: {exc}")
        return False


def dump_tensor_shape(tensor, output_path, name, meta=None, logger=None):
    """Append a tensor shape record to a JSONL file."""
    payload_meta = {"name": name, "kind": "tensor_shape"}
    if meta:
        payload_meta.update(meta)
    return dump_object_summary(
        tensor,
        output_path,
        meta=payload_meta,
        logger=logger,
    )
