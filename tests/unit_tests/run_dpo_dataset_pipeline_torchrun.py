import csv
import json
import logging
import os
import tempfile
import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import teletron.utils as teletron_utils
from teletron.datasets.dpo_dataset import WanDPODataset
from my_utils import DumpTensorIO


_ENCODER_UTILS = None


def _load_encoder_utils():
    global _ENCODER_UTILS
    if _ENCODER_UTILS is not None:
        return _ENCODER_UTILS
    override = os.environ.get("WAN_ENCODER_UTILS_PATH")
    if override:
        path = Path(override)
        if path.is_dir():
            path = path / "encoder_compare_utils.py"
    else:
        path = None
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "encoder_compare_utils.py"
            if candidate.exists():
                path = candidate
                break
    if path is None or not path.exists():
        raise RuntimeError("encoder_compare_utils.py not found; set WAN_ENCODER_UTILS_PATH")
    spec = importlib.util.spec_from_file_location("encoder_compare_utils", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load encoder_compare_utils.py from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ENCODER_UTILS = module
    return module


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


def _resolve_dump_rank():
    dump_rank = os.environ.get("WAN_DPO_DUMP_RANK")
    if dump_rank is None:
        env_local = os.environ.get("LOCAL_RANK")
        if env_local is not None:
            try:
                dump_rank = int(env_local)
            except ValueError:
                dump_rank = None
    if dump_rank is None:
        try:
            from megatron.core import mpu

            dump_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
        except TypeError:
            dump_rank = mpu.get_data_parallel_rank()
        except Exception:
            dump_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    return dump_rank


def _hash_file(path, chunk_size=1024 * 1024):
    if not path or not os.path.exists(path):
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_module_parameters(module):
    if module is None:
        return None
    hasher = hashlib.sha256()
    for name, param in sorted(module.named_parameters(), key=lambda x: x[0]):
        hasher.update(name.encode("utf-8"))
        hasher.update(str(param.dtype).encode("utf-8"))
        hasher.update(str(tuple(param.shape)).encode("utf-8"))
        data = param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        hasher.update(data)
    return hasher.hexdigest()


def _log_vae_hashes(encoder, encoder_config):
    if encoder is None:
        return
    vae = getattr(encoder, "vae", None)
    if vae is None:
        return
    vae_path = None
    if isinstance(encoder_config, dict):
        vae_cfg = encoder_config.get("vae", {})
        if isinstance(vae_cfg, dict):
            vae_path = vae_cfg.get("path")
    file_hash = _hash_file(vae_path)
    param_hash = _hash_module_parameters(vae)
    logging.info(
        "vae_hash path=%s file_sha256=%s param_sha256=%s",
        vae_path,
        file_hash,
        param_hash,
    )


def _build_encoder_batch(sample):
    if not isinstance(sample, dict):
        return None
    batch = dict(sample)
    images = batch.get("images")
    if not torch.is_tensor(images):
        return None
    if torch.is_tensor(images) and images.dim() == 4:
        batch["images"] = images.unsqueeze(0)
    return batch


def _build_image_vae_input(branch_item):
    if not isinstance(branch_item, dict):
        return None
    images = branch_item.get("images")
    if not torch.is_tensor(images):
        return None
    if images.dim() == 5:
        images = images[0]
    if images.dim() != 4:
        return None
    num_frames, channels, height, width = images.shape
    if num_frames < 1 or channels < 3:
        return None
    image = images[0].unsqueeze(0)
    end_image = branch_item.get("end_image")
    if torch.is_tensor(end_image):
        if end_image.dim() == 3:
            end_image = end_image.unsqueeze(0)
        elif end_image.dim() == 4 and end_image.shape[0] != 1:
            end_image = end_image[:1]
        if end_image.shape[-2:] != (height, width):
            end_image = None
    else:
        end_image = None
    if end_image is not None and num_frames >= 2:
        zeros = torch.zeros(3, num_frames - 2, height, width, device=image.device, dtype=image.dtype)
        vae_input = torch.concat([image.transpose(0, 1), zeros, end_image.transpose(0, 1)], dim=1)
    else:
        zeros = torch.zeros(3, num_frames - 1, height, width, device=image.device, dtype=image.dtype)
        vae_input = torch.concat([image.transpose(0, 1), zeros], dim=1)
    return vae_input


def _compare_encoder_outputs(sample, encoder, compare_rtol, compare_atol):
    if encoder is None or not isinstance(sample, dict):
        return
    try:
        from megatron.core import mpu

        cp_rank = mpu.get_tensor_context_parallel_rank()
    except Exception:
        cp_rank = 0
    if cp_rank != 0:
        return
    chosen = sample.get("chosen")
    rejected = sample.get("rejected")
    if not isinstance(chosen, dict) or not isinstance(rejected, dict):
        return
    chosen_batch = _build_encoder_batch(chosen)
    rejected_batch = _build_encoder_batch(rejected)
    if chosen_batch is None or rejected_batch is None:
        return
    raw_batch = {
        "chosen": chosen_batch,
        "rejected": rejected_batch,
    }
    # Run encoder forward to get embeddings/latents for compare.
    outputs = encoder.encode(raw_batch)
    if not isinstance(outputs, dict):
        return
    context = outputs.get("context")
    if torch.is_tensor(context):
        logging.info("encoder output dtype context=%s", context.dtype)
    tensor_dir_env = "WAN_DPO_EMBED_TENSOR_DIR" if os.environ.get("WAN_DPO_EMBED_TENSOR_DIR") else "WAN_DPO_PREVAE_TENSOR_DIR"
    dump_loader = DumpTensorIO(tensor_dir_env=tensor_dir_env)
    dump_rank = _resolve_dump_rank()
    logger = logging.getLogger(__name__)
    compare_map = {
        "context": ("prompt_emb", "context"),
        "img_emb_y": ("first_frame_emb", "y"),
        "latents": ("vae_latents", "input_latents"),
        "img_clip_feature": ("clip_feature", "clip_feature"),
    }
    for branch_name, branch_item in (("chosen", chosen), ("rejected", rejected)):
        pair_id = branch_item.get("dpo_pair_id")
        if pair_id is None:
            continue
        try:
            pair_id_value = int(pair_id) if not torch.is_tensor(pair_id) else int(pair_id.item())
        except (TypeError, ValueError, RuntimeError):
            continue
        branch_outputs = outputs.get(branch_name, {})
        img_emb_y = branch_outputs.get("img_emb_y")
        latents = branch_outputs.get("latents")
        if torch.is_tensor(img_emb_y):
            logging.info(
                "encoder output dtype pair_id=%s branch=%s img_emb_y=%s",
                pair_id_value,
                branch_name,
                img_emb_y.dtype,
            )
        if torch.is_tensor(latents):
            logging.info(
                "encoder output dtype pair_id=%s branch=%s latents=%s",
                pair_id_value,
                branch_name,
                latents.dtype,
            )
        for key, (tag_prefix, payload_key) in compare_map.items():
            if key == "context":
                actual = context
            else:
                actual = branch_outputs.get(key)
            if not torch.is_tensor(actual):
                continue
            tag = f"{tag_prefix}_{branch_name}"
            payload = dump_loader.load_tensors(pair_id_value, tag, dump_rank)
            if not isinstance(payload, dict) or payload_key not in payload:
                logger.info(
                    "skip encoder compare missing dump pair_id=%s tag=%s dump_rank=%s",
                    pair_id_value,
                    tag,
                    dump_rank,
                )
                continue
            expected = payload[payload_key]
            if not torch.is_tensor(expected):
                logger.info(
                    "skip encoder compare non-tensor dump pair_id=%s tag=%s dump_rank=%s",
                    pair_id_value,
                    tag,
                    dump_rank,
                )
                continue
            expected = expected.detach().cpu() if torch.is_tensor(expected) else expected
            actual_cpu = actual.detach().cpu()
            result = dump_loader.compare_tensors(expected, actual_cpu, rtol=compare_rtol, atol=compare_atol)
            dump_loader.write_compare_result(
                {
                    "tag": "encoder_compare",
                    "tensor": key,
                    "pair_id": pair_id_value,
                    "branch": branch_name,
                    "dump_rank": int(dump_rank) if dump_rank is not None else dump_rank,
                    "result": result,
                },
                torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            )
            logger.info(
                "encoder compare pair_id=%s branch=%s tensor=%s result=%s",
                pair_id_value,
                branch_name,
                key,
                result,
            )
        image_vae_input = _build_image_vae_input(branch_item)
        if torch.is_tensor(image_vae_input):
            tag = f"image_vae_input_{branch_name}"
            payload = dump_loader.load_tensors(pair_id_value, tag, dump_rank)
            if isinstance(payload, dict) and torch.is_tensor(payload.get("vae_input")):
                expected = payload["vae_input"].detach().cpu()
                actual = image_vae_input.detach().cpu()
                if expected.dtype != actual.dtype:
                    actual = actual.to(dtype=expected.dtype)
                result = dump_loader.compare_tensors(expected, actual, rtol=compare_rtol, atol=compare_atol)
                dump_loader.write_compare_result(
                    {
                        "tag": "image_vae_input_compare",
                        "pair_id": pair_id_value,
                        "branch": branch_name,
                        "dump_rank": int(dump_rank) if dump_rank is not None else dump_rank,
                        "result": result,
                    },
                    torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                )
                logger.info(
                    "image_vae_input compare pair_id=%s branch=%s result=%s",
                    pair_id_value,
                    branch_name,
                    result,
                )
            else:
                logger.info(
                    "skip image_vae_input compare missing dump pair_id=%s tag=%s dump_rank=%s",
                    pair_id_value,
                    tag,
                    dump_rank,
                )


def _log_encoder_dtypes(encoder):
    if encoder is None:
        return

    def _param_dtype(module):
        if module is None:
            return None
        try:
            return next(iter(module.parameters())).dtype
        except (StopIteration, AttributeError):
            return getattr(module, "dtype", None)

    logging.info(
        "encoder model dtype encoder=%s text=%s image=%s vae=%s",
        getattr(encoder, "dtype", None),
        _param_dtype(getattr(encoder, "text_encoder", None)),
        _param_dtype(getattr(encoder, "image_encoder", None)),
        _param_dtype(getattr(encoder, "vae", None)),
    )


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

    encoder_utils = _load_encoder_utils()
    default_config_path = Path(__file__).resolve().parents[2] / "examples" / "teleai" / "config" / "wan_dpo.py"
    encoder_config = encoder_utils.load_teletron_encoder_config(default_path=default_config_path)
    dataset_config = {
        "width": width,
        "height": height,
        "chosen_video_key": "chosen",
        "rejected_video_key": "rejected",
    }
    model_config = {"encoder": encoder_config} if encoder_config is not None else {}
    teletron_utils.set_config = lambda: {"dataset": dataset_config, "model_config": model_config}

    external_raw_dump = _resolve_external_dump(rank)
    external_prevae_dir = os.environ.get("WAN_DPO_PREVAE_TENSOR_DIR")
    use_external_dump = external_raw_dump is not None and external_prevae_dir

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        encoder = None
        encoder_compare_enabled = False
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
            encoder_compare_enabled = os.environ.get("WAN_DPO_ENCODER_COMPARE", "1") == "1"
            if encoder_compare_enabled:
                if not torch.cuda.is_available():
                    raise RuntimeError("encoder compare requires CUDA")
                if encoder_config is None or not isinstance(encoder_config, dict):
                    raise RuntimeError("encoder config is missing; set WAN_DPO_ENCODER_CONFIG_PATH")
                encoder_name = encoder_config.get("type")
                if not encoder_name:
                    raise RuntimeError("encoder config missing type field")
                # Build and initialize the Teletron encoder used for compare.
                encoder = encoder_utils.build_teletron_encoder(encoder_config, device=torch.cuda.current_device())
                _log_encoder_dtypes(encoder)
                _log_vae_hashes(encoder, encoder_config)
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
                sample = dataset[record_idx]
                if encoder_compare_enabled and encoder is not None:
                    compare_rtol = float(os.environ.get("WAN_DPO_PREVAE_COMPARE_RTOL", "1e-5"))
                    compare_atol = float(os.environ.get("WAN_DPO_PREVAE_COMPARE_ATOL", "1e-8"))
                    _compare_encoder_outputs(sample, encoder, compare_rtol, compare_atol)

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
