from typing import List
import numpy as np
from .base_dataset import BaseDataset
from teleai_data_tool.schema.clip import ImageWithCaption, Clip
from teleai_data_tool.file.lmdb_client import LmdbClient
from teleai_data_tool.file.file_client import FileClient
import json
from teleai_data_tool.logger import logger
from tqdm import tqdm
from cattrs import structure
import random, math

class VariableMixDataset(BaseDataset):
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
        
        self.image_lmdb_client = LmdbClient(data_type="image")
        self.image_file_client = FileClient(data_type="image")
        self.video_file_client = FileClient()
        self.video_lmdb_client = LmdbClient()

    def load_data_list(self) -> List[dict]:
        data_list = {
            "image_list": [],
            "video_list": [],
        }
        if self.data_path_list.get("image_list", None):
            for data_path in tqdm(self.data_path_list["image_list"]):
                with open(data_path) as f:
                    dataset = json.load(f)
                for clip in dataset["images"]:
                    clip = structure(clip, ImageWithCaption)
                    clip.file_path = f"{dataset['image_data_root']}{clip.image.file_name}"
                    clip.meta["data_format"] = dataset["image_data_type"]
                    setattr(clip, "data_type", "image")
                    data_list["image_list"].append(clip)

        if self.data_path_list.get("video_list", None):
            for data_path in tqdm(self.data_path_list["video_list"]):
                with open(data_path) as f:
                    dataset = json.load(f)
                for clip in dataset["clips"]:
                    clip = structure(clip, Clip)
                    clip.file_path = f"{dataset['clip_data_root']}:{clip.file_path}"
                    clip.meta["data_format"] = dataset["clip_data_type"]
                    setattr(clip, "data_type", "video")
                    data_list["video_list"].append(clip)
        return data_list
    
    def filter_image_data(self,):
        filter_cfg = self.filter_cfg.get("image_filter_cfg", dict())

        image_aesthetic_th = filter_cfg.get("aesthetic_th", 4)
        image_watermark_th = filter_cfg.get("watermark_th", 1.0)
        image_unsafe_th = filter_cfg.get("unsafe_th", 1.0)
        image_area_th = filter_cfg.get("area_th", 256*256)
        
        # fileter tag 
        too_small = 0
        aes_mismatch = 0
        watermark_mismatch = 0
        unsafe_mismatch = 0

        valid_data_list = []
        for clip in self.data_list["image_list"]:
            if clip.image.height * clip.image.width < image_area_th:
                too_small += 1
                continue
            if clip.filter_state is not None:
                # aesthetic
                if (
                    clip.filter_state.aesthetic is not None
                    and clip.filter_state.aesthetic < image_aesthetic_th
                ):
                    aes_mismatch += 1
                    continue
                # watermark 
                if (
                    clip.filter_state.water_mark is not None
                    and clip.filter_state.water_mark > image_watermark_th
                ):
                    watermark_mismatch += 1
                    continue
                # unsafe
                if (
                    clip.filter_state.unsafe is not None
                    and clip.filter_state.unsafe > image_unsafe_th
                ):
                    unsafe_mismatch += 1
                    continue
            valid_data_list.append(clip)
        logger.info(
            f"finish filter image dataset, from {len(self.data_list['image_list'])} to {len(valid_data_list)} \n"
            f"too small data {too_small} \n"
            f"aesthetic mismatch data {aes_mismatch} \n"
            f"watermark mismatch data {watermark_mismatch} \n "
            f"unsafe mismatch data {unsafe_mismatch} \n"
        )
        return valid_data_list

    def filter_video_data(self,):
        filter_cfg = self.filter_cfg.get("video_filter_cfg", dict())
        
        optical_flow_th = filter_cfg.get("optical_flow_th", 2)
        aesthetic_th = filter_cfg.get("aesthetic_th", 4)
        motion_th = filter_cfg.get("motion_th", 0) 
        clearity_th = filter_cfg.get("clearity_th", 0.8) 
        laplacian_th = filter_cfg.get("laplacian_th", 0)
        training_suitability_th = filter_cfg.get("training_suitability_th", 3.7) 
        area_th = filter_cfg.get("area_th", 720*480) 
        min_frames = filter_cfg.get("min_frames", 33)
        fps = filter_cfg.get("fps", 16)

        # fileter tag 
        too_small = 0
        too_short = 0
        aes_mismatch = 0
        motion_mismatch = 0
        clearity_mismatch = 0
        suitability_mismatch = 0

        valid_data_list = []
        for clip in self.data_list["video_list"]:
            frame_interval = int(max(1.0, math.ceil(clip.fps / fps)))
            min_frames_clip = int(min_frames*frame_interval)
            setattr(clip, "frame_interval", frame_interval)
            setattr(clip, "min_num_frames", min_frames_clip)
            if clip.height * clip.width < area_th:
                too_small += 1
                continue
            if clip.length < min_frames_clip:
                too_short += 1
                continue

            if clip.filter_state is not None:
                # aesthetic
                if (
                    clip.filter_state.aesthetic is None
                    or clip.filter_state.aesthetic < aesthetic_th
                ):
                    aes_mismatch += 1
                    continue

                # laplacian, 部分数据没有laplacian，所以这里是 and
                if (
                    clip.filter_state.laplacian is not None
                    and clip.filter_state.laplacian < laplacian_th
                ):
                    clearity_mismatch += 1
                    continue

                # optical_flow
                if clip.filter_state.optical_flow != -1.0:
                    if (
                        clip.filter_state.optical_flow is None
                        or clip.filter_state.optical_flow < optical_flow_th
                    ):
                        motion_mismatch += 1
                        continue
            
                # clearity
                if (
                    clip.filter_state.clearity is not None
                    and clip.filter_state.clearity < clearity_th
                ):
                    clearity_mismatch += 1
                    continue

                # motion
                if (
                    clip.filter_state.motion is not None
                    and clip.filter_state.motion < motion_th
                ):
                    motion_mismatch += 1
                    continue

                # training_suitability
                if (
                    clip.filter_state.video_training_suitability is not None
                    and clip.filter_state.video_training_suitability < training_suitability_th
                ):
                    suitability_mismatch += 1
                    continue
            valid_data_list.append(clip)
        logger.info(
            f"finish filter video dataset, from {len(self.data_list['video_list'])} to {len(valid_data_list)} \n"
            f"too small data {too_small} \n"
            f"too short data {too_short} \n"
            f"motion mismatch data {motion_mismatch} \n"
            f"aesthetic mismatch data {aes_mismatch} \n"
            f"clearity score mismatch data {clearity_mismatch} \n"
            f"suitability score mismatch data {suitability_mismatch} \n"
        )
        return valid_data_list

    def filter_data(self):
        valid_data_list = self.filter_image_data() + self.filter_video_data()
        logger.info(
            f"finish filter dataset, from {len(self.data_list['video_list']+self.data_list['image_list'])} to {len(valid_data_list)} \n"
        )
        return valid_data_list
    
    def get_image_data_info(self, clip):
        data_dict = dict()
        if clip.meta["data_format"] == "lmdb":
            video = self.image_lmdb_client.get(clip.file_path)
        elif clip.meta["data_format"] == "file":
            video = self.image_file_client.get(clip.file_path)
        data_dict["clip_info"] = clip
        data_dict["video"] = video
        data_dict["video_info"] = clip.video_info
        data_dict["video_height"] = clip.image.height
        data_dict["video_width"] = clip.image.width
        return data_dict

    def get_video_data_info(self, clip):
        data_dict = dict()
        if clip.meta["data_format"] == "lmdb":
            video = self.video_lmdb_client.get(clip.file_path, num_threads=8)
        elif clip.meta["data_format"] == "file":
            video = self.video_file_client.get(clip.file_path, num_threads=8)
        data_dict["clip_info"] = clip
        data_dict["video"] = video
        data_dict["video_info"] = clip.video_info
        data_dict["video_length"] = clip.length
        data_dict["video_height"] = clip.height
        data_dict["video_width"] = clip.width
        data_dict["slice_index"] = None
        if len(clip.caption.frame_range) > 0:
            last_slice = clip.caption.frame_range[-1]
            slice_length = len(clip.caption.frame_range)
            min_num_frames = int(clip.video_info[0]*clip.frame_interval)
            if (last_slice[1] - last_slice[0]) < min_num_frames:
                slice_length = slice_length-1
            slice_index = random.randint(0, max(0, slice_length-1))
            data_dict["video_valid_range"] = clip.caption.frame_range[slice_index]
            data_dict["slice_index"] = slice_index
        else:
            data_dict["video_valid_range"] = clip.valid_range
        data_dict["fps"] = clip.fps
        data_dict["frame_interval"] = clip.frame_interval
        return data_dict

    def get_data_info(self, idx):
        clip = super().get_data_info(idx)
        if clip.data_type == 'image':
            data_dict = self.get_image_data_info(clip)
        elif clip.data_type == 'video':
            data_dict = self.get_video_data_info(clip)
        return data_dict