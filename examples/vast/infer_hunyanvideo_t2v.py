import torch
from pipelines import HunyuanVideoPipeline
from diffusers.utils import export_to_video

width, height, = 1280, 720
num_frames = 49
num_inference_steps = 10
device = torch.device("cuda:0")
base_model_path = "ckpt/hunyuan/hunyuanvideo_13b"
transformer_model_path = 'ckpt/checkpoint_epoch_1_step_50000/transformer'
prompt = "A woman is crouching in front of the oven in the kitchen, holding the oven door handle with both hands and opening the oven door."

pipeline = HunyuanVideoPipeline.from_pretrained(
    base_model_path,
    transformer_model_path=transformer_model_path,
    torch_dtype=torch.bfloat16,
).to(device)
pipeline.vae.enable_tiling()

print(f"prompt: {prompt}")
video = pipeline(
    prompt=prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=num_inference_steps,
    seed=42,
    model_type='t2v',
).frames[0]

export_to_video(video, "result.mp4", fps=15)