from collections import OrderedDict
from typing import Dict, List, Tuple
import math 

ASPECT_RATIO = {
    # "1:1": 1.0,
    # "4:3": 4/3,
    "16:9": 832/480,
    "9:16": 480/832
}


class Bucket:
    def __init__(self, bucket_config: Dict[str, Dict[str, int]], padding_size: int = 16):
        """
        Args:
            bucket_config (Dict[str, Dict[str, int]]):
            eg. "256px":{
                "1": {
                    "bsz": 64,
                    "prob": 0.5,
                }, # nframe: {batch_size, resolution sampled prob}
            }
            padding_size (int): The height and width of images will be padded to be divisible by this number.
        """
        self.padding_size = padding_size
        self.hw_criteria = get_hw_criteria(bucket_config)
        self.resolution = get_resolution(self.hw_criteria, padding_size)
        self.bucket_bsz = get_bucket_bsz(bucket_config, self.resolution)
        self.bucket_prob = get_bucket_prob(bucket_config)

def get_hw_criteria(bucket_config: Dict[str, Dict[int, int]]) -> Dict[str, int]:
    """
    Args:

        bucket_config (Dict[str, Dict[int, int]]):
    Returns:
        
        Dict[str, int]: {"1x256px": 256x256}
    """
    hw_criteria_dict = {}   # {"1x256px": 256x256}
    for resolution_type, resolution_info in bucket_config.items():
        for nframe, _ in resolution_info.items():
            hw_criteria_dict[f'{nframe}x{resolution_type}'] = get_hw_criteria_area(resolution_type)
    return hw_criteria_dict

def get_hw_criteria_area(resolution_type: str) -> int:
    if "x" in resolution_type:
        width, height = int(resolution_type.split('x')[0]), int(resolution_type.split('x')[1])
        return width * height
    if "px" in resolution_type:
        resolution = int(resolution_type.replace("px", ""))
        return resolution * resolution
    else:
        raise ValueError("resolution type must be xxxpx (e.g. 960px, 720px, 480px).")

def get_resolution(hw_criteria: Dict[str, int], padding_size: int) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    Args:
        
        hw_criteria (Dict[str, int]): {"1x256px": 256x256}

        padding_size (int): The height and width of images will be padded to be divisible by this number.
    Returns:
        
        Dict[str, Dict[str, Tuple[int, int]]]: {"1x256px": {"1:1": (256, 256)}}
    """
    resolution = dict()
    for time_space_info, area in hw_criteria.items():
        resolution[time_space_info] = dict()
        for aspect_ratio_type, aspect_ratio in ASPECT_RATIO.items():
            height, width = get_resolution_with_aspect_ratio(area, aspect_ratio, padding_size)
            resolution[time_space_info][aspect_ratio_type] = (height, width)
    return resolution

def get_resolution_with_aspect_ratio(area: int, aspect_ratio: float, padding_size: int):
    width = math.sqrt(area*aspect_ratio)
    height = width / aspect_ratio

    width = round(width / padding_size) * padding_size
    height = round(height / padding_size) * padding_size
    return int(height), int(width)

def get_bucket_bsz(bucket_config: Dict[str, Dict[int, int]], resolution: Dict[str, Dict[str, Tuple[int, int]]]):
    """
    Args:
        
        bucket_config (Dict[str, Dict[int, int]]): {"256px":{"1"\: {"bsz": 64, "prob": 0.5}}}

        resolution (Dict[str, Tuple[int, int]]]): {"1x256px": {"1:1": (256, 256)}}

    Returns:
        
        Dict[str, int]: {"1x256x256": 64}
    """
    bucket_bs = dict()
    for resolution_type, resolution_info in bucket_config.items():
        for nframe, info in resolution_info.items():
            for height, width in resolution[f'{nframe}x{resolution_type}'].values():
                bucket_bs[f'{nframe}x{height}x{width}'] = info['bsz']
    return bucket_bs

def get_bucket_prob(bucket_config: Dict[str, Dict[int, int]]) -> Dict[str, float]:
    bucket_prob = dict()
    for resolution_type, resolution_info in bucket_config.items():
        for nframe, info in resolution_info.items():
            bucket_prob[f'{nframe}x{resolution_type}'] = info['prob']
    return bucket_prob

def get_random_resolution_type(bucket_prob: Dict[str, float], random_prob: float, length: int):
    """
    Args:
        
        bucket_prob (Dict[str, float]): {"1x256px": 0.5, "1x512px": 0.5}

        random_prob (float): [0, 1)
    Returns:
        
        str: "1x256px"
    """
    prefix=str(length)+'x'
    bucket_prob = {k: v for k, v in bucket_prob.items() if k.startswith(prefix)}
    total_prob = sum(bucket_prob.values())
    rand_prob = random_prob * total_prob
    for key, value in bucket_prob.items():
        if rand_prob < value:
            return key
        rand_prob -= value
    raise ValueError("Invalid random_prob.")

def get_close_resolution_type(area: int, hw_criteria: Dict[str, int]):
    for key, value in hw_criteria.items():
        if area > value:
            return key
    return None

def get_close_ratio(height: int, width: int):
    aspect_ratio = float(height) / float(width)
    closest_ratio = min(
        ASPECT_RATIO.keys(), key=lambda ratio: abs(aspect_ratio - ASPECT_RATIO[ratio])
    )
    return closest_ratio
