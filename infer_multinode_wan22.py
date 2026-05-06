import torch
import os
from PIL import Image
from typing import Dict, List, Tuple
from pipelines import WanVideoI2VMoEPipeline
from diffusers.utils import export_to_video
from pathlib import Path

from prompts.HardPrompt import PROMPT_CONFIGS
from prompts.i2v150Prompt import PROMPT_CONFIGS
from prompts.arena_i2v_and_150 import PROMPT_CONFIGS
# from prompts.ContinueBmk_add import PROMPT_CONFIGS
from prompts.flf2vPrompt import PROMPT_CONFIGS
from prompts.vfx_fl2v_prompt import PROMPT_CONFIGS

# from prompts.arena_i2v import PROMPT_CONFIGS
# from prompts.arena_i2v_samples import PROMPT_CONFIGSzuijiaaSSASDDSFS
# from prompts.continue_t2v_first import PROMPT_CONFIGS


LOW_NOISE_CKPT_PATH = os.environ["LOW_NOISE_CKPT_PATH"]
HIGH_NOISE_CKPT_PATH = os.environ["HIGH_NOISE_CKPT_PATH"]
SAVEDIR = os.environ.get("INFER_SAVEDIR", "./results/wan22_infer")
WAN2_1_I2V_14B_480P = os.environ["WAN2_1_I2V_14B_480P"]

# 使用环境变量获取节点信息
NODE_INDEX = int(os.environ.get('GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX', 0))
NODE_NUM = int(os.environ.get('GEMINI_TASKS_NUM', 1))

# 每个节点使用8个GPU
GPU_IDS = list(range(8))
# GPU_IDS = []

DEFAULT_CONFIG = {
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "highnoise_cfg_scale": 5,
    "lownoise_cfg_scale": 5,
    "num_inference_step": 50,
    "save_fps": 8,
    "negative_prompt":"完全黑暗，完全静止，剧烈抖动，扭曲，不真实，色调艳丽，过曝，静态，细节模糊不清，字幕，静止，整体发灰，最差质量，低质量，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿",
    "flow_shift": 5,
    "latent_frame": 31,
}

PIPELINE_CONFIG = dict(
    model_config=dict(
        low_noise_model=dict(
            path=LOW_NOISE_CKPT_PATH, # ema_model.pt
            config=dict(
                has_image_input=False, # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=36, # t2v:16 i2v:36
                dim=5120, # 1.3B:1536 10B:5120 14B:5120
                ffn_dim=13824, # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=40, # 1.3B:12 10B:40 14B:40
                num_layers=40, # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=False, 
            ),
        ),
        high_noise_model=dict(
            path=HIGH_NOISE_CKPT_PATH, # ema_model.pt
            config=dict(
                has_image_input=False, # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=36, # t2v:16 i2v:36
                dim=5120, # 1.3B:1536 10B:5120 14B:5120
                ffn_dim=13824, # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=40, # 1.3B:12 10B:40 14B:40
                num_layers=40, # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=False, 
            ),
        ),
        encoder=dict(
            vae=dict(
                type="TeleaiVideoVAE_2_1",
                path=os.path.join(WAN2_1_I2V_14B_480P, "Wan2.1_VAE.pth"),
                tiler_kwargs=dict(
                    tiled=False,
                    tile_size=(34, 34),
                    tile_stride=(18, 16),
                ),
            ),
            text_encoder=dict(
                path=os.path.join(WAN2_1_I2V_14B_480P, "models_t5_umt5-xxl-enc-bf16.pth"),
                tokenizer_path=os.path.join(WAN2_1_I2V_14B_480P, "google/umt5-xxl"),
            ),
            image_encoder=dict(
                path=os.path.join(WAN2_1_I2V_14B_480P, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            ),
        ),
    ),
    torch_dtype=torch.bfloat16,
    device="cuda",
    boundary=900,
)

class InferenceConfig:
    """集中管理推理配置参数"""

    def __init__(
        self,
        prompt: str,
        ref_images: Dict[int, str],
        height: int = DEFAULT_CONFIG["height"],
        width: int = DEFAULT_CONFIG["width"],
        num_frames: int = DEFAULT_CONFIG["num_frames"],
        lownoise_cfg_scale: float = DEFAULT_CONFIG["lownoise_cfg_scale"],
        highnoise_cfg_scale: float = DEFAULT_CONFIG["highnoise_cfg_scale"],
        num_inference_step: int = DEFAULT_CONFIG["num_inference_step"],
        save_fps: int = DEFAULT_CONFIG["save_fps"],
        negative_prompt: str = DEFAULT_CONFIG["negative_prompt"],
        flow_shift: int = DEFAULT_CONFIG["flow_shift"],
        latent_frame: int = DEFAULT_CONFIG["latent_frame"],
    ):
        self.prompt = prompt
        self.ref_images = ref_images
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.highnoise_cfg_scale = highnoise_cfg_scale
        self.lownoise_cfg_scale = lownoise_cfg_scale
        self.num_inference_step = num_inference_step
        self.save_fps = save_fps
        self.negative_prompt = negative_prompt
        self.flow_shift = flow_shift
        self.latent_frame = latent_frame

def generate_video_filename(
    result_name: str, 
    config: InferenceConfig
) -> str:
    
    """生成视频文件名"""
    width, height = config.width, config.height
    base_name = f"{result_name}_{height}x{width}_f{config.num_frames}_s{config.num_inference_step}_fps{config.save_fps}_hg{config.highnoise_cfg_scale}_lg{config.lownoise_cfg_scale}_shift{config.flow_shift}"
    return f"{base_name}.mp4"

def prepare_reference_images(config: InferenceConfig) -> List[Image.Image]:
    ref_images = []
    for frame_id, img_path in config.ref_images.items():
        original_img = Image.open(img_path).convert("RGB")
        ref_images.append((frame_id, original_img))
    return ref_images

def load_pipeline(
    device: torch.device,
) -> WanVideoI2VMoEPipeline:
    pipe = WanVideoI2VMoEPipeline(PIPELINE_CONFIG)
    pipe.enable_cpu_offload()
    return pipe


def inference_worker(
    rank: int,
    world_size: int,
    inference_configs: List[Tuple[str, InferenceConfig]],
    gpu_ids: List[int],
):
    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    # 分配模型处理任务
    infer_task = inference_configs[rank::world_size]
    if not infer_task:
        print(f"[Rank {rank}] No task")
        return
    
    pipe = load_pipeline(device)
    for test_name, config in infer_task:
        try:
            save_dir = Path(SAVEDIR)
            save_dir.mkdir(parents=True, exist_ok=True)

            save_name = generate_video_filename(test_name, config)
            save_path = save_dir / save_name

            if save_path.exists():
                print(f"[Rank {rank}] 文件已存在，跳过: {save_path}")
                continue
                            
            ref_images = prepare_reference_images(config)
            # 执行推理
            result = pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                ref_images=ref_images,
                height=config.height,
                width=config.width,
                num_frames=config.num_frames,
                highnoise_cfg_scale=config.highnoise_cfg_scale,
                lownoise_cfg_scale=config.lownoise_cfg_scale,
                num_inference_steps=config.num_inference_step,
                seed=42,
                sigma_shift=config.flow_shift,
            )

            export_to_video(result, str(save_path), fps=config.save_fps, quality=6)            
            print(f"[Rank {rank}] 保存成功: {save_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Rank {rank}] 处理任务 {test_name} 时出错: {str(e)}")
    

def distribute_tasks_across_nodes(prompt_configs: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """将任务分配到各个节点"""
    config_items = list(prompt_configs.items())[::-1]
    total_tasks = len(config_items)
    
    # 计算每个节点分配的任务
    tasks_per_node = total_tasks // NODE_NUM
    remainder = total_tasks % NODE_NUM
    
    # 计算当前节点的任务范围
    start_index = NODE_INDEX * tasks_per_node + min(NODE_INDEX, remainder)
    end_index = start_index + tasks_per_node + (1 if NODE_INDEX < remainder else 0)
    
    node_tasks = dict(config_items[start_index:end_index])
    
    print(f"[Node {NODE_INDEX}] 分配任务: {start_index}-{end_index-1} (共{len(node_tasks)}个任务)")
    print(f"[Node {NODE_INDEX}] 任务列表: {list(node_tasks.keys())}")
    
    return node_tasks


def run_inference_pipeline(
    prompt_configs: Dict[str, Dict[str, str]],
    gpu_ids: List[int] = None,
):
    """运行完整推理流程（优化后版本）"""
    # GPU配置
    available_gpus = torch.cuda.device_count()
    gpu_ids = gpu_ids or list(range(available_gpus))
    world_size = len(gpu_ids)

    # 准备所有推理配置
    inference_config_list = []
    for test_name, config_data in prompt_configs.items():
        inference_config = InferenceConfig(
            prompt=config_data["dense_prompt"],
            ref_images=config_data["ref_images"],
            **{k: v for k, v in DEFAULT_CONFIG.items()},
        )
        inference_config_list.append((test_name, inference_config))

    # 启动单次多进程推理
    print(f"\n{'='*40}\n节点 {NODE_INDEX} 开始处理 {len(inference_config_list)} 个测试用例\n{'='*40}")
    import torch.multiprocessing as mp
    from functools import partial

    mp.spawn(
        partial(
            inference_worker,
            world_size=world_size,
            inference_configs=inference_config_list,
            gpu_ids=gpu_ids,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    try:
        print(f"节点配置: 当前节点索引={NODE_INDEX}, 总节点数={NODE_NUM}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        
        # 将任务分配到当前节点
        node_prompt_configs = distribute_tasks_across_nodes(PROMPT_CONFIGS)
        
        if not node_prompt_configs:
            print(f"[Node {NODE_INDEX}] 没有分配到任务，退出")
            exit(0)
            
        run_inference_pipeline(
            prompt_configs=node_prompt_configs,
            gpu_ids=GPU_IDS,
        )
        
        print(f"[Node {NODE_INDEX}] 所有任务处理完成")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"主程序运行出错: {str(e)}")