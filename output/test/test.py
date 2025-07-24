import torch
import glob


file_paths = glob.glob("../*__fsdp_wrapped_module.blocks.17._checkpoint_wrapped_module._fsdp_wrapped_module.self_attn.o_input.pt")
#/nvfile-heatstorage/teleai-infra/wxe/Dancegrpo_verl/output/0__fsdp_wrapped_module.blocks.0._checkpoint_wrapped_module._fsdp_wrapped_module.self_attn.q_input.pt
#/nvfile-heatstorage/teleai-infra/wxe/Dancegrpo_verl/output/0__fsdp_wrapped_module.blocks.0._checkpoint_wrapped_module_input.pt
#/nvfile-heatstorage/teleai-infra/wxe/Dancegrpo_verl/output/0__fsdp_wrapped_module.blocks.0._checkpoint_wrapped_module._fsdp_wrapped_module_input.pt
#/nvfile-heatstorage/teleai-infra/wxe/Dancegrpo_verl/output/0__fsdp_wrapped_module.blocks.0._checkpoint_wrapped_module._fsdp_wrapped_module.self_attn.k_input.pt
all_inputs = []

for path in sorted(file_paths):
    data = torch.load(path)

    print(f"Loaded {path}:")
    print("  name:", data["name"])
    print("  input shape:", data["input"][0].shape if isinstance(data["input"], tuple) else type(data["input"]))
    print("  output shape:", data["output"].shape if hasattr(data["output"], 'shape') else type(data["output"]))
    
    all_inputs.append(data["output"][0])
    print(data["input"][0].type())

data_all = torch.stack(all_inputs).squeeze()

# 匹配所有 *_self_attn.o_input.pt 文件

file_paths = glob.glob("/nvfile-heatstorage/teleai-infra/wxe/dancegrpo_aigc/output/0__fsdp_wrapped_module.blocks.17._checkpoint_wrapped_module._fsdp_wrapped_module.self_attn.o_input.pt")

all_inputs = []

for path in sorted(file_paths):
    data = torch.load(path)

    
    print(f"Loaded {path}:")
    print("  name:", data["name"])
    print("  input shape:", data["input"][0].shape if isinstance(data["input"], tuple) else type(data["input"]))
    print("  output shape:", data["output"].shape if hasattr(data["output"], 'shape') else type(data["output"]))
    
    all_inputs.append(data["output"][0])
    print(data["input"][0].type())

all_inputs_new = torch.stack(all_inputs).squeeze()

# print(data_all.shape)
import hashlib

def tensor_md5(tensor):
    tensor = tensor.float()  # 将张量转换为 float32 类型
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()

print(tensor_md5(data_all))
print(tensor_md5(all_inputs_new))
data_all_norm=data_all.float().norm().item()
data_all_new_norm=all_inputs_new.float().norm().item()
print("Overall L2 norm:", data_all_new_norm,data_all_norm)
