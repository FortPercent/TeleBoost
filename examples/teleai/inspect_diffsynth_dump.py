import argparse
import os
import glob
import torch


def _empty_reason(value):
    if value is None:
        return "none"
    if isinstance(value, str) and value == "":
        return "empty_string"
    if torch.is_tensor(value):
        if value.numel() == 0:
            return "empty_tensor"
        return None
    if isinstance(value, dict):
        if len(value) == 0:
            return "empty_dict"
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "empty_list"
        return None
    return None


def _walk(value, path, empties):
    reason = _empty_reason(value)
    if reason is not None:
        empties.append((path, reason, type(value).__name__))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_str = str(key)
            child_path = f"{path}.{key_str}" if path else key_str
            _walk(item, child_path, empties)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            child_path = f"{path}[{idx}]"
            _walk(item, child_path, empties)


def _iter_paths(target):
    if os.path.isdir(target):
        pattern = os.path.join(target, "*.pt")
        for path in sorted(glob.glob(pattern)):
            yield path
        return
    if any(ch in target for ch in ["*", "?", "[", "]"]):
        for path in sorted(glob.glob(target)):
            yield path
        return
    yield target


def inspect_file(path):
    payload = torch.load(path, weights_only=False, map_location="cpu")
    empties = []
    _walk(payload, "", empties)
    return empties


def main():
    parser = argparse.ArgumentParser(description="Inspect DiffSynth dump .pt files for empty values.")
    parser.add_argument(
        "--path",
        required=True,
        help="Path to a .pt file, a directory, or a glob pattern.",
    )
    args = parser.parse_args()

    paths = list(_iter_paths(args.path))
    if not paths:
        raise SystemExit(f"No files found for: {args.path}")

    for path in paths:
        empties = inspect_file(path)
        print(f"== {path} ==")
        if not empties:
            print("no empty values found")
            continue
        for item_path, reason, type_name in empties:
            display_path = item_path if item_path else "<root>"
            print(f"{display_path}: {reason} ({type_name})")


if __name__ == "__main__":
    main()
