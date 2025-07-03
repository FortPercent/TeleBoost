import torch
from PIL import Image
from typing import Dict, List, Tuple
from teletron.models.teleai.models.dit.teleai_dit.model_manager import ModelManager
from pipeline import TeleaiVideoPipeline
from diffusers.utils import export_to_video
from pathlib import Path
import torch.multiprocessing as mp
from functools import partial
from I2VPrompt import PROMPT_CONFIGS
import os
import traceback

DEFAULT_CONFIG = {
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    "cfg_scale": 5.0,
    "num_inference_step": 50,
    "tile": False,
    "save_fps": 16,
    "negative_prompt":"色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿",
    "flow_shift": 5
}
SAVEDIR = "tele_expr3_2x9b_step900_fp32"
GPU_IDS = [7]# 4, 5, 6, 7]
MODEL_OFFLOAD = True

class InferenceConfig:
    """集中管理推理配置参数"""

    def __init__(
        self,
        prompt: str,
        ref_images: Dict[int, str],
        height: int = DEFAULT_CONFIG["height"],
        width: int = DEFAULT_CONFIG["width"],
        num_frames: int = DEFAULT_CONFIG["num_frames"],
        cfg_scale: float = DEFAULT_CONFIG["cfg_scale"],
        num_inference_step: int = DEFAULT_CONFIG["num_inference_step"],
        tile: bool = DEFAULT_CONFIG["tile"],
        save_fps: int = DEFAULT_CONFIG["save_fps"],
        negative_prompt: str = DEFAULT_CONFIG["negative_prompt"],
        flow_shift: int = DEFAULT_CONFIG["flow_shift"]
    ):
        self.prompt = prompt
        self.ref_images = ref_images
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.cfg_scale = cfg_scale
        self.num_inference_step = num_inference_step
        self.tile = tile
        self.save_fps = save_fps
        self.negative_prompt = negative_prompt
        self.flow_shift = flow_shift

def generate_video_filename(
    result_name: str, 
    config: InferenceConfig
) -> str:
    
    """生成视频文件名"""
    width, height = config.width, config.height
    base_name = f"{result_name}_{height}x{width}_f{config.num_frames}_s{config.num_inference_step}_fps{config.save_fps}_g{config.cfg_scale}_shift{config.flow_shift}"
    return f"{base_name}.mp4"

def prepare_reference_images(config: InferenceConfig) -> List[Image.Image]:
    ref_images = []
    is_vertical = []
    for _, img_path in config.ref_images.items():
        original_img = Image.open(img_path).convert("RGB")
        ref_images.append(original_img)
        is_vertical.append(original_img.height > original_img.width)
    return ref_images, is_vertical

def load_pipeline(
    device: torch.device, moe_config
) -> TeleaiVideoPipeline:
    
    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        ["/nvfile-heatstorage/myk/vast/dense_models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32, # Image Encoder is loaded with float32
    )
    model_manager.load_models(
        [
            "/nvfile-heatstorage/myk/vast/dense_models/models_t5_umt5-xxl-enc-bf16.pth",
            "/nvfile-heatstorage/myk/vast/dense_models/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.float32,
    )

    model_list = []
    for _, model_info in moe_config.items():
        model_list.append(model_info["dit_path"])
    
    model_manager.load_models(
        model_list,
        torch_dtype=torch.bfloat16,
        device="cpu" if MODEL_OFFLOAD else "cuda"
    )
    pipe = TeleaiVideoPipeline.load_moe_model_from_model_manager(
            model_manager, 
            moe_config=moe_config,
            torch_dtype=torch.bfloat16, 
            device=device,
            model_offload=MODEL_OFFLOAD
        )
    return pipe


def inference_worker(
    rank: int,
    world_size: int,
    inference_configs: List[Tuple[str, InferenceConfig]],
    gpu_ids: List[int],
    moe_config: Dict
):
    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    # 分配模型处理任务
    infer_task = inference_configs[rank::world_size]
    if not infer_task:
        print(f"[Rank {rank}] No task")
        return
    
    pipe = load_pipeline(device, moe_config)
    for test_name, config in infer_task:
        try:
            save_dir = Path(__file__).parent / "results" / SAVEDIR
            save_dir.mkdir(parents=True, exist_ok=True)

            save_name = generate_video_filename(test_name, config)
            save_path = save_dir / save_name

            if os.path.exists(save_path):
                continue
           
            # 准备参考图像
            ref_images, is_vertical = prepare_reference_images(config)
            # 执行推理
            output = pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                input_image=ref_images[0],
                height=config.width if is_vertical[0] else config.height,
                width=config.height if is_vertical[0] else config.width,
                num_frames=config.num_frames,
                cfg_scale=config.cfg_scale,
                num_inference_steps=config.num_inference_step,
                seed=42,
                tiled=config.tile,
                sigma_shift=config.flow_shift,
                verbose=False,
            )
            export_to_video(output, str(save_path), fps=config.save_fps)
            print(f"[Rank {rank}] 保存成功: {save_path}")

        except Exception:
            print(f"[Rank {rank}] 处理任务 {test_name} 时出错: {traceback.format_exc()}")
    
def run_inference_pipeline(
    prompt_configs: Dict[str, Dict[str, str]],
    gpu_ids: List[int] = None, moe_config: Dict = None 
):
    """运行完整推理流程（优化后版本）"""

    # 准备所有推理配置
    inference_config_list = []
    for test_name, config_data in prompt_configs.items():
        inference_config = InferenceConfig(
            prompt=config_data["short_prompt"],
            ref_images=config_data["ref_images"],
            **{k: v for k, v in DEFAULT_CONFIG.items()},
        )
        inference_config_list.append((test_name, inference_config))

    # 启动单次多进程推理
    print(f"\n{'='*40}\n开始批量处理所有测试用例\n{'='*40}")
    world_size = len(gpu_ids)
    #inference_worker(rank=0, world_size=1, inference_configs=inference_config_list, gpu_ids=gpu_ids, moe_config=moe_config)
    mp.spawn(
        partial(
            inference_worker,
            world_size=world_size,
            inference_configs=inference_config_list,
            gpu_ids=gpu_ids,
            moe_config=moe_config
        ),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    from argparse import ArgumentParser 
    parser = ArgumentParser()
    parser.add_argument("--models", nargs='+', help='Path of moe model component weights, from high noise to low noise')
    parser.add_argument("--timesteps", nargs='+', type=int, help="Timesteps to switch model, from high to low")
    # TODO: support variable num layers
    args = parser.parse_args()
    if len(args.models) < 2 or len(args.timesteps) < 2:
        print("TeleAI model requires at least 2 model components")
        exit()
    moe_config = {
        "model1": {"dit_path": args.models[0], "timestep_range": (args.timesteps[1], args.timesteps[0])}
    }
    for i in range(1, len(args.models)-1):
        moe_config.update({
            f"model{i+1}": {"dit_path": args.models[i], "timestep_range": (args.timesteps[i+1], args.timesteps[i])}
        })
    moe_config.update({
        f"model{len(args.models)}": {"dit_path": args.models[-1], "timestep_range": (args.timesteps[-1], args.timesteps[-2])}
    }) 
    

    # try:
    run_inference_pipeline(
        prompt_configs=PROMPT_CONFIGS,
        gpu_ids=GPU_IDS,
        moe_config=moe_config
    )
    # except Exception as e:
    #     print(f"主程序运行出错: {str(e)}")