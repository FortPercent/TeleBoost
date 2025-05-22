import os
import torch
from vast.models import  HunyuanVideoTransformer3DModel
import transformers
import diffusers

print("transformers version:", transformers.__version__)
print("diffusers version:", diffusers.__version__)

ckpt_path = "/nvfile-heatstorage/hyc/vast/work_dirs/hunyuanvideo_i2vhy_sp2_720p_85_24fps_0512/models/checkpoint_epoch_1_step_5000"
bin_path = f"{ckpt_path}/transformer/diffusion_pytorch_model.bin"
path = f"{ckpt_path}/transformer"
save_path = f"{ckpt_path}/transformer_safetensor"

print(f"check .bin file exist: {bin_path}")
assert os.path.exists(bin_path), f"{bin_path} not exist!"

transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    path,
    allow_pickle=True,
    trust_remote_code=True,   
    use_safetensors=False
)
transformer.save_pretrained(save_path)