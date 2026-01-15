#!/usr/bin/env python3
"""
独立的Qwen奖励计算服务
在cuda:6,7上运行，监听视频文件并计算奖励
"""

import os
import sys
import torch

# 【关键修正】设置CUDA设备（必须在import torch之前）
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 【修正】直接导入你已经写好的函数
from train_grpo_wan_fsdp_new import compute_qwen_vllm_reward_on_cuda7

def setup_distributed(rank, world_size):
    """
    设置分布式训练环境
    """
    os.environ['MASTER_ADDR'] = 'localhost'  # 修正拼写错误
    os.environ['MASTER_PORT'] = '12355'
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    #dist.destroy_process_group()
    pass

def run_qwen_service(rank, world_size):
    """
    启动Qwen奖励计算服务
    """
    #setup_distributed(rank, world_size)
    qwen_model_path = "/gemini/space/Qwen/Qwen2___5-VL-72B-Instruct"
    from vllm import LLM, SamplingParams
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.90)
    print(f"Loading Qwen model from {qwen_model_path}")
    videos_dir = "/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward/videos"
    my_llm = LLM(model = qwen_model_path, tensor_parallel_size = 4, gpu_memory_utilization = 0.90, allowed_local_media_path = videos_dir,)
    try:
        compute_qwen_vllm_reward_on_cuda7(
            llm=my_llm,
            sampling_params=sampling_params,
            reward_dir="/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward",
            qwen_model_path="/gemini/space/Qwen/Qwen2___5-VL-72B-Instruct",
            device="auto",
            check_interval=1.0,
            max_wait_time=360000000000000.0,
        )
    except Exception as e:
        print(f"❌ Error in Qwen service: {e}")
    finally:
        #cleanup_distributed()
        pass
if __name__ == "__main__":
    print("🚀 Starting Qwen reward computation service on cuda:6,7...")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        exit(1)
    run_qwen_service(0,1)
    # # 启动服务
    # world_size = 2 
    # print(f"Using {world_size} GPUs for distributed training")  # 修正拼写错误
    # mp.spawn(run_qwen_service, args=(world_size,), nprocs=world_size, join=True)