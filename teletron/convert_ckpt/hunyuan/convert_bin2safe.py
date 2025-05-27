import os
import sys
import json
import torch
from vast.models import  HunyuanVideoTransformer3DModel
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/vast

ckpt_name = "hunyuanvideo_i2vhy_token_replace"
ckpt_name = "hunyuanvideo_i2v_multimask"

# read from baseline config
config_path = "/nvfile-heatstorage/yxy/code/Teletron/model_paths.json"
with open(config_path, 'r') as f:
    config = json.load(f)
ckpt_path = config.get(ckpt_name)

# custom set ckpt_path
ckpt_path = "/nvfile-heatstorage/hyc/vast/work_dirs/hunyuanvideo_i2vhy_sp2_720p_85_24fps_0512/models/checkpoint_epoch_1_step_5000/teletron/"
bin_path = f"{ckpt_path}/transformer"
# exported more than one .bin file, so we need to merge them into one .bin file
bin_file_list = [f"{bin_path}/diffusion_pytorch_model-00001-of-00002.bin", f"{bin_path}/diffusion_pytorch_model-00002-of-00002.bin"]
save_path = f"{ckpt_path}/transformer_safetensor"

# print(f"check .bin file exist: {bin_path}")
# assert os.path.exists(bin_file), f"{bin_file} not exist!"

state_dict_all = {}
for bin_file in bin_file_list:
    state_dict = torch.load(bin_file)
    state_dict_all.update(state_dict)
torch.save(state_dict_all, f"{bin_path}/diffusion_pytorch_model.bin")

transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    bin_path,
    allow_pickle=True,
    trust_remote_code=True,   
    use_safetensors=False
)
transformer.save_pretrained(save_path)
print(f"save safetensors model to {save_path}")