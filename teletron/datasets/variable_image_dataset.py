from typing import List
import numpy as np
from .base_dataset import BaseDataset
from teleai_data_tool.schema.clip import ImageWithCaption
from teleai_data_tool.file.lmdb_client import LmdbClient
from teleai_data_tool.file.file_client import FileClient
import json
from teleai_data_tool.logger import logger
from tqdm import tqdm
from cattrs import structure

class VariableImageDataset(BaseDataset):
    def __init__(
        self,
        data_path_list,
        transforms,
        filter_cfg=dict(),
        data_weight_list=[],
        serialize_data=False,
    ) -> None:
        self.data_path_list = data_path_list
        self.data_weight_list = data_weight_list
        super().__init__(
            ann_file="",
            serialize_data=serialize_data,
            test_mode=False,
            lazy_init=False,
            max_refetch=1,
            pipeline=transforms,
            filter_cfg=filter_cfg,
        )
        
        self.lmdb_client = LmdbClient(data_type="image")
        self.file_client = FileClient(data_type="image")

    def load_data_list(self) -> List[dict]:
        data_list = []
        for data_path in tqdm(self.data_path_list):
            with open(data_path) as f:
                dataset = json.load(f)
            for clip in dataset["images"]:
                clip = structure(clip, ImageWithCaption)
                clip.file_path = f"{dataset['image_data_root']}{clip.image.file_name}"
                clip.meta["data_format"] = dataset["image_data_type"]
                data_list.append(clip)
        return data_list

    def filter_data(self):
        aesthetic_th = self.filter_cfg.get("aesthetic_th", 4)
        watermark_th = self.filter_cfg.get("watermark_th", 0.3)
        unsafe_th = self.filter_cfg.get("unsafe_th", 0.4)
        area_th = self.filter_cfg.get("area_th", 720*480)

        # fileter tag 
        too_small = 0
        aes_mismatch = 0
        watermark_mismatch = 0
        unsafe_mismatch = 0

        valid_data_list = []
        for clip in self.data_list:
            if clip.image.height * clip.image.width < area_th:
                too_small += 1
                continue
            if clip.filter_state is not None:
                # aesthetic
                if (
                    clip.filter_state.aesthetic is not None
                    and clip.filter_state.aesthetic < aesthetic_th
                ):
                    aes_mismatch += 1
                    continue
                # watermark 
                if (
                    clip.filter_state.water_mark is not None
                    and clip.filter_state.water_mark > watermark_th
                ):
                    watermark_mismatch += 1
                    continue
                # unsafe
                if (
                    clip.filter_state.unsafe is not None
                    and clip.filter_state.unsafe > unsafe_th
                ):
                    unsafe_mismatch += 1
                    continue
            valid_data_list.append(clip)

        logger.info(
            f"finish filter dataset, from {len(self.data_list)} to {len(valid_data_list)} \n"
            f"too small data {too_small} \n"
            f"aesthetic mismatch data {aes_mismatch} \n"
            f"watermark mismatch data {watermark_mismatch} \n "
            f"unsafe mismatch data {unsafe_mismatch} \n"
        )
        return valid_data_list

    def get_data_info(self, idx):
        clip: ImageWithCaption = super().get_data_info(idx)
        data_dict = dict()
        if clip.meta["data_format"] == "lmdb":
            video = self.lmdb_client.get(clip.file_path)
        elif clip.meta["data_format"] == "file":
            video = self.file_client.get(clip.file_path)
        data_dict["clip_info"] = clip
        data_dict["video"] = video
        data_dict["video_info"] = clip.video_info
        data_dict["video_height"] = clip.image.height
        data_dict["video_width"] = clip.image.width
        return data_dict
