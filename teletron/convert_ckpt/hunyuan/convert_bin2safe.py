import os
import sys
import json
from vast.models import  HunyuanVideoTransformer3DModel
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/vast

ckpt_name = "hunyuanvideo_i2vhy_token_replace"
ckpt_name = "hunyuanvideo_i2v_multimask"

config_path = "/nvfile-heatstorage/yxy/code/Teletron/model_paths.json"
with open(config_path, 'r') as f:
    config = json.load(f)
ckpt_path = config.get(ckpt_name)

bin_path = f"{ckpt_path}/transformer"
bin_file = f"{bin_path}/diffusion_pytorch_model.bin"
save_path = f"{ckpt_path}/transformer_safetensor"

print(f"check .bin file exist: {bin_path}")
assert os.path.exists(bin_file), f"{bin_file} not exist!"

print(f"load bin model from {bin_path}")
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    bin_path,
    allow_pickle=True,
    trust_remote_code=True,   
    use_safetensors=False
)
transformer.save_pretrained(save_path)
print(f"save safetensors model to {save_path}")