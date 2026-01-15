import torch
import torch
from torch.distributed import tensor as dtensor  # 导入 DTensor 相关模块

# 假设 rank_0 到 rank_7 的文件都在同一个目录
input_dir = "/root/verl/ckpts/DAPO/DAPO-Qwen2.5-32B/global_step_200/actor"
output_file = "/root/verl/ckpts/DAPO/DAPO-Qwen2.5-32B/global_step_200/actordiffusion_pytorch_model.bin"

import torch
from torch.distributed import tensor as dtensor  # 导入 DTensor 相关模块

# 读取 rank 权重
state_dict = {}
for rank in range(8):
    print(f"{input_dir}/model_world_size_8_rank_{rank}.pt")
    shard = torch.load(f"{input_dir}/model_world_size_8_rank_{rank}.pt", map_location="cpu", weights_only=False)
    # 合并 shard 到 state_dict
    for key, value in shard.items():
        if key not in state_dict:
            state_dict[key] = []
        state_dict[key].append(value)

# 如果是分片 tensor，要手动 cat / merge
for key in state_dict:
    if isinstance(state_dict[key][0], torch.Tensor):
        # 将分片的 Tensor 沿维度 0 拼接
        try:
            state_dict[key] = torch.cat(state_dict[key], dim=0)
        except RuntimeError as e:
            # 捕获拼接失败的异常（例如形状不匹配）
            print(f"Failed to merge tensor for key '{key}': {e}")

# 保存合并后的模型
torch.save(state_dict, output_file)
print("模型已合并保存为 merged_model.pth")