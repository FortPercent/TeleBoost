import unittest
import os
import sys
import shutil
import tempfile
import json
import torch
import numpy as np
import imageio.v3 as iio
from PIL import Image
from torchvision.transforms.functional import to_pil_image

# ==========================================
# 1. 环境 Mock (模拟缺失的依赖库，确保脚本可运行)
# ==========================================
try:
    from teleai_data_tool.file.file_client import FileClient
    from teleai_data_tool.file.lmdb_client import LmdbClient
except ImportError:
    # 简单的 Mock，让 Dataset 代码能跑通
    class FileClient:
        def get(self, path): return path # 直接返回路径
    class LmdbClient:
        def get(self, path): return path

try:
    from megatron.core import mpu
except ImportError:
    class MockMPU:
        def get_data_parallel_rank(self, with_context_parallel=False): return 0
    mpu = MockMPU()

# 模拟 logger
def get_global_logger():
    import logging
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    return logger

# 注入到 modules 中，防止你的 Dataset 导入报错
sys.modules["teleai_data_tool.file.file_client"] = type("m", (), {"FileClient": FileClient})
sys.modules["teleai_data_tool.file.lmdb_client"] = type("m", (), {"LmdbClient": LmdbClient})
sys.modules["megatron.core"] = type("m", (), {"mpu": mpu})
sys.modules["my_utils"] = type("m", (), {"get_global_logger": get_global_logger})

# ==========================================
# 2. 导入你的 Dataset 定义 
# (在实际使用中，这里应该是 from your_project import WanDPODataset)
# 为了测试独立性，这里假设你上面提供的代码已经保存为 wandpo_dataset.py
# 或者你可以直接把那一大段 Dataset 代码粘贴在这个位置。
# ==========================================
# 假设你把代码保存在当前目录的 wandpo_code.py 中
# from wandpo_code import WanDPODataset 

# --- 为了方便演示，我必须在这里重新定义一遍关键的 Dataset 类，
# --- 实际运行时，请删除下面的定义，直接 import 你的类 ---
# (此处省略大量的类定义粘贴，假设上下文已有，
#  但在运行脚本时，你需要把上面的 Dataset 代码都粘贴到这里，
#  或者确保它可以被 import)

# ... [请在此处插入或 import 你的 WanDPODataset 及相关类] ...
# 为了让这个回复简洁，我假设环境里已经有了 WanDPODataset
# 如果没有，请把你的代码贴在下面:

# (为了脚本能独立运行，这里我写一个最小化的 Placeholder，
#  你需要用真实的类替换这里)
# ---------------------------------------------------------------------------
# ⚠️⚠️⚠️ 真实测试时，请删除下面这几行 Mock，使用真实的 Dataset 代码 ⚠️⚠️⚠️

from teletron.datasets.dpo_dataset import WanDPODataset
# ---------------------------------------------------------------------------


# ==========================================
# 3. 核心修复逻辑：反归一化与转换
# ==========================================
def _raw_image_to_pil_smart(raw_image):
    """
    智能转换函数：能处理 [-1, 1], [0, 1], [0, 255] 以及各种维度
    """
    if isinstance(raw_image, Image.Image):
        return raw_image
    
    if torch.is_tensor(raw_image):
        tensor = raw_image.detach().cpu().float()
        
        # 1. 自动剥离维度 (Batch, Time, etc.)
        # 目标是拿到 [C, H, W]
        while tensor.dim() > 3:
            # 如果第一维是 1，挤压掉
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            else:
                # 可能是 Batch 或 Time 维度，取第一个元素
                tensor = tensor[0]
        
        # 此时 tensor 应该是 [C, H, W]
        # 2. 判断是否需要反归一化
        # 如果最小值 < 0 (通常是 -1)，说明是 [-1, 1]
        if tensor.min() < 0:
            print(f"[Convert] Detected [-1, 1] range (min={tensor.min():.2f}), denormalizing...")
            tensor = tensor * 0.5 + 0.5 # 变回 [0, 1]
        
        # 3. 如果是 [0, 1] (float)，转为 [0, 255] (uint8)
        if tensor.max() <= 1.05:
            tensor = tensor * 255.0
            
        tensor = tensor.clamp(0, 255).to(torch.uint8)
        return to_pil_image(tensor)
        
    return raw_image


# ==========================================
# 4. 单元测试用例
# ==========================================
class TestWanDPOPipeline(unittest.TestCase):
    
    def test_load_real_data_and_check_image(self):
        print(">>> 正在初始化 Dataset (使用真实配置)...")
        
        # 1. 这里填入你真实的路径配置
        real_data_path_list = [
            "/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/out_parts/prompt_video_pairs_matched_image.part0.csv",
            # 为了测试速度，建议先只测 part0，如果没问题再把下面解除注释
            # "/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/out_parts/prompt_video_pairs_matched_image.part1.csv",
            # ... 其他 part ...
        ]
        
        # 2. 实例化 Dataset
        dataset = WanDPODataset(
            transforms=None,
            
            # === 这里填入你的配置 ===
            dataset_base_path="", # 你配置里是空字符串
            data_path_list=real_data_path_list, # 指定上面的列表
            dataset_repeat=1, # 测试时建议设为 1，不用设为 2
            
            # ⚠️ 关键：请确认 CSV 里的列名是否叫 'chosen' 和 'rejected'
            # 如果 CSV 里叫 'video_path' 或其他名字，这里必须改！
            chosen_video_key="chosen",   
            rejected_video_key="rejected",
            
            # 其他参数保持默认或根据需要修改
            height=480,  # 你的配置参数
            width=832,   # 你的配置参数
            num_frames=49,
            dataset_metadata_path=None # 既然用了 data_path_list，这个可以为 None
        )
        
        print(f">>> Dataset 加载完成，总样本数: {len(dataset)}")
        
        if len(dataset) == 0:
            print("❌ 错误: Dataset 为空！请检查 CSV 文件路径是否存在，或文件是否为空。")
            return

        # 3. 读取第 0 个样本
        print(">>> 正在读取第 0 个样本 (这将触发 LoadVideo 和 ToTensor)...")
        try:
            sample = dataset[0]
        except Exception as e:
            print(f"❌ 读取失败: {e}")
            # 如果是 CSV 列名不对，通常会报 KeyError
            return

        # 4. 获取 Video Tensor
        # 假设你的 dataset 结构里，key 是 chosen/rejected
        if "chosen" not in sample:
            print(f"❌ 样本中没找到 'chosen' key。现有的 keys: {sample.keys()}")
            return
            
        video_tensor = sample["chosen"]["video"]
        print(f"[Check] Video Tensor Shape: {video_tensor.shape}")
        
        # 5. 提取并还原首帧 (使用我们之前讨论的逻辑)
        # Tensor 形状通常是 [B, C, T, H, W] 或 [C, T, H, W]
        # 假设 batch=1 (from repeat/unsqueeze) -> [1, 3, 49, 480, 832]
        
        first_frame_tensor = None
        
        # 鲁棒的维度提取逻辑
        if video_tensor.dim() == 5: # B C T H W
            first_frame_tensor = video_tensor[0, :, 0, :, :]
        elif video_tensor.dim() == 4: # C T H W
            first_frame_tensor = video_tensor[:, 0, :, :]
            
        if first_frame_tensor is None:
            print(f"❌ 无法识别 Tensor 维度: {video_tensor.shape}")
            return
            
        print(f"[Check] First Frame Tensor Shape: {first_frame_tensor.shape}")

        # 6. 转换回 PIL 并保存
        # (确保 _raw_image_to_pil_smart 函数已定义在脚本中)
        try:
            restored_img = _raw_image_to_pil_smart(first_frame_tensor)
            
            save_path = "./debug_real_data_frame.png"
            restored_img.save(save_path)
            print(f"✅ [成功] 首帧已保存至: {os.path.abspath(save_path)}")
            print(f"    图像模式: {restored_img.mode}, 尺寸: {restored_img.size}")
            print("    请打开图片检查颜色是否正常！")
            
        except Exception as e:
            print(f"❌ 图片还原失败: {e}")

if __name__ == "__main__":
    unittest.main()