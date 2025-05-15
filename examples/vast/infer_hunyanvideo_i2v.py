import torch
from pipelines import HunyuanVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image
from typing import List, Dict
from torchvision.transforms import InterpolationMode, functional as F


def prepare_reference_images(width: int, height: int, num_frames: int, ref_frames: Dict[int, str]) -> List[Image.Image]:
    """准备参考图像"""
    ref_images = [Image.new("RGB", (width, height), (0, 0, 0)) for _ in range(num_frames)]
    for frame_idx, img_path in ref_frames.items():
        if frame_idx >= num_frames:
            raise ValueError(f"参考帧索引 {frame_idx} 超过总帧数 {num_frames}")
       
        original_img = Image.open(img_path)
        resized_img = F.resize(original_img, (height, width), InterpolationMode.BILINEAR)
        ref_images[frame_idx] = resized_img
    return ref_images


width, height, = 1280, 720
num_frames = 49
num_inference_steps = 50
base_model_path = "ckpt/hunyuan/hunyuanvideo_13b"
transformer_model_path = 'ckpt/checkpoint_epoch_1_step_50000/transformer'
ref_frames = {0: "sample/image/oven.jpg"}
prompt = "A woman is crouching in front of the oven in the kitchen, holding the oven door handle with both hands and opening the oven door"
device = torch.device("cuda:0")

ref_images = prepare_reference_images(width, height, num_frames, ref_frames)
pipeline = HunyuanVideoPipeline.from_pretrained(
    base_model_path,
    transformer_model_path=transformer_model_path,
    torch_dtype=torch.bfloat16,
).to(device)
pipeline.vae.enable_tiling()

video = pipeline(
    prompt=prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=num_inference_steps,
    seed=42,
    model_type='i2v',
    ref_images=ref_images,
    guidance_scale=4.0,
    embedded_guidance_scale=1.0,
).frames[0]

export_to_video(video, "oven.mp4", fps=15)