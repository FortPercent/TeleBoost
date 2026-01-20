import csv
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import teletron.utils as teletron_utils
from teletron.datasets.dpo_dataset import WanDPODataset


def _write_image(path, color, size):
    image = Image.fromarray(np.full((size[1], size[0], 3), color, dtype=np.uint8))
    image.save(path)


def _init_dist_from_torchrun():
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available")
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    if rank_env is None or world_size_env is None:
        raise RuntimeError("run with torchrun so RANK/WORLD_SIZE are set")
    rank = int(rank_env)
    world_size = int(world_size_env)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    return rank, world_size


def _resolve_ranked_path(path, rank):
    if "{rank}" in path:
        return Path(path.format(rank=rank))
    raw_path = Path(path)
    if raw_path.exists():
        return raw_path
    root = raw_path.with_suffix("")
    return Path(f"{root}_rank{rank}{raw_path.suffix}")


def _resolve_external_dump(rank):
    raw_path_env = os.environ.get("WAN_DPO_DATASET_DUMP_FILE")
    if not raw_path_env:
        return None
    path = _resolve_ranked_path(raw_path_env, rank)
    if path.exists():
        return path
    return None


def _load_raw_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("tag") != "dataset.raw":
                continue
            stage = record.get("stage")
            if stage is not None and stage != "raw":
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            dump_id = record.get("dump_id")
            data_id = record.get("data_id")
            records.append((dump_id, data_id, payload))
    records.sort(key=lambda x: x[0] if x[0] is not None else -1)
    return records


def _run_pipeline(rank):
    height = 480
    width = 832
    num_frames = 49
    max_pixels = 400000

    teletron_utils.set_config = lambda: {"dataset": {"width": width, "height": height}}

    external_raw_dump = _resolve_external_dump(rank)
    external_prevae_dir = os.environ.get("WAN_DPO_PREVAE_TENSOR_DIR")
    use_external_dump = external_raw_dump is not None and external_prevae_dir

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        if use_external_dump:
            os.environ["WAN_DPO_PREVAE_COMPARE"] = "1"
            os.environ.setdefault("WAN_DPO_PREVAE_COMPARE_RTOL", "1e-5")
            os.environ.setdefault("WAN_DPO_PREVAE_COMPARE_ATOL", "1e-8")
            os.environ["WAN_DPO_PREVAE_COMPARE_FILE"] = str(
                tmp_path / f"prevae_compare_rank{rank}.jsonl"
            )
            dataset_base_path = os.environ.get("WAN_DPO_DATASET_BASE_PATH", "")
            metadata_path = external_raw_dump
            part_files = []
            ref_path = None
            part_prompt = None
            if not dataset_base_path:
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            record = json.loads(line)
                            payload = record.get("payload", {}) if isinstance(record, dict) else {}
                            chosen = payload.get("chosen")
                            if isinstance(chosen, str) and not os.path.isabs(chosen):
                                raise RuntimeError(
                                    "WAN_DPO_DATASET_BASE_PATH is required for relative paths"
                                )
                            break
                except FileNotFoundError:
                    raise RuntimeError(
                        f"raw dataset dump not found: {metadata_path}"
                    )
        else:
            chosen_path = tmp_path / f"chosen_rank{rank}.png"
            rejected_path = tmp_path / f"rejected_rank{rank}.png"
            ref_path = tmp_path / f"ref_rank{rank}.png"
            _write_image(chosen_path, 64, (64, 64))
            _write_image(rejected_path, 192, (64, 64))
            _write_image(ref_path, 128, (64, 64))

            full_prompt = f"full prompt rank {rank}"
            part_prompt = f"part0 prompt rank {rank}"

            os.environ["WAN_DPO_PREVAE_COMPARE"] = "0"
            os.environ.pop("WAN_DPO_DATASET_DUMP_FILE", None)

            metadata_path = tmp_path / "prompt_video_pairs_matched_image.csv"
            part_files = [
                tmp_path / f"prompt_video_pairs_matched_image.part{i}.csv" for i in range(8)
            ]
            with open(metadata_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["chosen", "rejected", "prompt", "input_image", "dpo_pair_id"])
                writer.writerow(
                    [
                        chosen_path.name,
                        rejected_path.name,
                        full_prompt,
                        ref_path.name,
                        rank,
                    ]
                )
            for path in part_files:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["chosen", "rejected", "prompt", "input_image", "dpo_pair_id"])
                    writer.writerow(
                        [
                            chosen_path.name,
                            rejected_path.name,
                            part_prompt,
                            ref_path.name,
                            rank,
                        ]
                    )
            dataset_base_path = str(tmp_path)

        transforms = [
            {
                "type": "InjectRawFirstImageFromVideo",
                "video_key": "video",
                "output_key": "raw_first_image",
            },
            {
                "type": "CompareImageEmbedFromVideo",
                "video_key": "video",
                "compare_input": True,
                "compare_end": False,
                "torch_dtype": "bfloat16",
                "min_value": -1,
                "max_value": 1,
            },
            {
                "type": "PreprocessVideoToTensor",
                "input_key": "video",
                "output_key": "video",
                "torch_dtype": "bfloat16",
                "pattern": "B C T H W",
                "min_value": -1,
                "max_value": 1,
                "skip_if_tensor": True,
            },
            {
                "type": "InjectImagesFromVideoTensor",
                "video_key": "video",
                "output_key": "images",
            },
            {
                "type": "InjectPromptToTopLevel",
                "prompt_key": "prompt",
            },
            {
                "type": "PackInputsNoResize",
                "normalize": False,
                "image_keys": ["images"],
                "embedding_keys": ["raw_first_image", "input_image"],
            },
        ]

        if use_external_dump:
            raw_records = _load_raw_records(metadata_path)
            if not raw_records:
                raise AssertionError("external dump mode requires raw dataset records")
            dataset = WanDPODataset(
                transforms=transforms,
                dataset_base_path=dataset_base_path,
                dataset_metadata_path=str(metadata_path),
                data_path_list=None,
                chosen_video_key="chosen",
                rejected_video_key="rejected",
                height=height,
                width=width,
                num_frames=num_frames,
                max_pixels=max_pixels,
                time_division_factor=4,
                time_division_remainder=1,
                height_division_factor=16,
                width_division_factor=16,
                dataset_repeat=1,
            )
            if len(dataset) != len(raw_records):
                raise AssertionError(
                    f"raw record count mismatch rank={rank} dataset_len={len(dataset)} "
                    f"raw_len={len(raw_records)}"
                )
            compare_file = Path(os.environ["WAN_DPO_PREVAE_COMPARE_FILE"])
            for record_idx, (_, record_data_id, raw_entry) in enumerate(raw_records):
                if record_data_id is None:
                    raise AssertionError("external dump mode requires data_id for compare")
                prompt_value = raw_entry.get("prompt") if isinstance(raw_entry, dict) else None
                if prompt_value is not None:
                    logging.info(
                        "compare prompt rank=%s data_id=%s idx=%s prompt=%s",
                        rank,
                        record_data_id,
                        record_idx,
                        prompt_value,
                    )
                _ = dataset[record_idx]

            cp_rank = 0
            try:
                from megatron.core import mpu
                cp_rank = mpu.get_tensor_context_parallel_rank()
            except Exception:
                cp_rank = 0
            if cp_rank == 0:
                if not compare_file.exists():
                    raise AssertionError(f"compare output missing: {compare_file}")
                results = []
                with open(compare_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                if not results:
                    raise AssertionError("compare output is empty")
                failures = [
                    r for r in results
                    if not (isinstance(r, dict) and isinstance(r.get("result"), dict) and r["result"].get("allclose"))
                ]
                logging.info(
                    "compare summary rank=%s total=%s ok=%s fail=%s",
                    rank,
                    len(results),
                    len(results) - len(failures),
                    len(failures),
                )
                if failures:
                    sample = failures[0]
                    raise AssertionError(f"compare failed: {sample}")
            else:
                logging.info("skip compare summary on cp_rank=%s", cp_rank)
        else:
            dataset = WanDPODataset(
                transforms=transforms,
                dataset_base_path=dataset_base_path,
                dataset_metadata_path=str(metadata_path),
                data_path_list=[str(path) for path in part_files] if part_files else None,
                chosen_video_key="chosen",
                rejected_video_key="rejected",
                height=height,
                width=width,
                num_frames=num_frames,
                max_pixels=max_pixels,
                time_division_factor=4,
                time_division_remainder=1,
                height_division_factor=16,
                width_division_factor=16,
                dataset_repeat=1,
            )

            data_id = 0
            sample = dataset[data_id]
            assert set(sample.keys()) == {"chosen", "rejected"}

            for branch in ("chosen", "rejected"):
                item = sample[branch]
                assert item["images"].shape == (1, 3, height, width)
                assert item["images"].dtype == torch.bfloat16
                assert item["raw_first_image"].shape == (1, 3, height, width)
                assert item["raw_first_image"].dtype == torch.uint8
                if ref_path is not None:
                    assert item["input_image"] == ref_path.name
                assert item["frame_interval"] == 1
                if part_prompt is not None:
                    assert item["short_prompt"] == [part_prompt]
                    assert item["dense_prompt"] == [part_prompt]
                    assert item["struct_prompt"] == [part_prompt]
                assert torch.max(item["images"]).item() <= 1.001
                assert torch.min(item["images"]).item() >= -1.001


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    rank, world_size = _init_dist_from_torchrun()
    logging.info("torchrun rank=%s world_size=%s", rank, world_size)
    try:
        _run_pipeline(rank)
    except Exception:
        logging.exception("dpo dataset pipeline test failed on rank %s", rank)
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
