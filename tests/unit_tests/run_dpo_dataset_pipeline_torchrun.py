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


def _write_image(path, color):
    image = Image.fromarray(np.full((32, 32, 3), color, dtype=np.uint8))
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


def _run_pipeline(rank):
    height = 32
    width = 32

    teletron_utils.set_config = lambda: {"dataset": {"width": width, "height": height}}

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        chosen_path = tmp_path / f"chosen_rank{rank}.png"
        rejected_path = tmp_path / f"rejected_rank{rank}.png"
        ref_path = tmp_path / f"ref_rank{rank}.png"
        _write_image(chosen_path, 64)
        _write_image(rejected_path, 192)
        _write_image(ref_path, 128)

        prompt = f"unit test prompt rank {rank}"
        metadata_path = tmp_path / f"meta_rank{rank}.jsonl"
        metadata_path.write_text(
            json.dumps(
                {
                    "chosen": chosen_path.name,
                    "rejected": rejected_path.name,
                    "prompt": prompt,
                    "input_image": ref_path.name,
                    "dpo_pair_id": rank,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        transforms = [
            {
                "type": "InjectRawFirstImageFromVideo",
                "video_key": "video",
                "output_key": "raw_first_image",
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

        dataset = WanDPODataset(
            transforms=transforms,
            dataset_base_path=str(tmp_path),
            dataset_metadata_path=str(metadata_path),
            chosen_video_key="chosen",
            rejected_video_key="rejected",
            height=height,
            width=width,
            num_frames=1,
            max_pixels=height * width,
            dataset_repeat=1,
        )

        sample = dataset[0]
        assert set(sample.keys()) == {"chosen", "rejected"}

        for branch in ("chosen", "rejected"):
            item = sample[branch]
            assert item["images"].shape == (1, 3, height, width)
            assert item["images"].dtype == torch.bfloat16
            assert item["raw_first_image"].shape == (1, 3, height, width)
            assert item["raw_first_image"].dtype == torch.uint8
            assert item["input_image"] == ref_path.name
            assert item["frame_interval"] == 1
            assert item["short_prompt"] == [prompt]
            assert item["dense_prompt"] == [prompt]
            assert item["struct_prompt"] == [prompt]
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
