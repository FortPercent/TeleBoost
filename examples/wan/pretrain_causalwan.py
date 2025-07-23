import argparse
import os
from omegaconf import OmegaConf
import torch
from teletron.train import parse_args
from teletron.train.causal_trainer import CausalTrainer
from teletron.models.flow_match import FlowMatchScheduler
from teletron.train.utils import get_batch, loss_func
from megatron.core import mpu

def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    group.add_argument("--no_save", action="store_false")
    group.add_argument("--load_raw_video", action="store_false")
    group.add_argument("--gradient-checkpointing", action="store_false")
    group.add_argument("--real-name", type=str, default="Wan2.1-T2V-14B")
    group.add_argument("--negative_prompt",type=str,
                       default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    # group.add_argument("--base_paths", type=str, nargs = '+',
    #                    default=['/nvfile-heatstorage/teleai-infra/kaikai/HumanData_subset_500/merged_videos_latents',]
    #                    )
    # group.add_argument("--metadata_paths", type=str, nargs = '+',
    #                    default=['/nvfile-heatstorage/teleai-infra/kaikai/HumanData_subset_500/filtered_500.csv',]
    #                    )
    # group.add_argument("--logdir", type=str, default="wan_experiments_test", help="Path to the directory to save logs")
    # group.add_argument("--test_valid", type=str, default="")
    # group.add_argument("--moe-step-factor-list", type=float, action='append')
    # group = parser.add_argument_group(title='encoder args')
    # group.add_argument("--encoder_model_path", type=str, nargs = '+',default=
    #                    ['/workspace/Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth', 
    #                     '/workspace/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth', 
    #                     '/workspace/Wan2___1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth']
    #                    )
    # group.add_argument("--encoder_tokenizer_path", type=str, default=
    #                    "/workspace/Wan2___1-I2V-14B-480P/google/umt5-xxl")
    
    return parser

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config_path", type=str, required=True)
#     # parser.add_argument("--save", action="store_true")
#     parser.add_argument("--save", type=str, default="wan_experiments_test", help="Path to the directory to save logs")

#     args = parser.parse_args()

#     config = OmegaConf.load(args.config_path)
#     default_config = OmegaConf.load("/nvfile-heatstorage/teleai-infra/kaikai/dreamingforcing/WorldVideo/configs/default_config.yaml")
#     config = OmegaConf.merge(default_config, config)
#     trainer = DiffusionTrainer(config)
#     # breakpoint()
#     trainer.train()




import torch.distributed as dist
import debugpy

def wait_for_debugger(rank_to_debug=0, port=5678):
    rank = int(os.environ.get("RANK", "0"))
    # All ranks pause here before debugger
    if rank == rank_to_debug:
        print(f"[Rank {rank}] Waiting for debugger on port {port}...")
        debugpy.listen(("0.0.0.0", port))
        debugpy.wait_for_client()
        print(f"[Rank {rank}] Debugger attached.")

def forward_step(data_iterator, model):
    flow_scheduler = model.scheduler
    prompt_emb = {}
    batch = next(data_iterator)
    clean_latent = batch["latents"]
    
    noise = torch.randn_like(clean_latent) if "noise" not in batch else batch["noise"]
    batch_size, num_frame = clean_latent.shape[:2]
    index = model._get_timestep(
        0,
        flow_scheduler.num_train_timesteps,
        clean_latent.shape[0],
        clean_latent.shape[1],
        model.num_frame_per_block,
        uniform_timestep=False
    )
    timestep = flow_scheduler.timesteps[index].to(dtype=model.dtype, device=model.device)
    
    def broadcast_timesteps(input: torch.Tensor):
        tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
        if mpu.get_tensor_context_parallel_world_size() > 1:
            dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())

    broadcast_timesteps(timestep)
    broadcast_timesteps(noise)
    prompt_emb["context"] = batch["context"]
    prompt_emb["unconditional_dict"]= batch["unconditional_dict"] if "unconditional_dict"  in batch else None
    training_target = flow_scheduler.training_target(clean_latent, noise, timestep)
    
    noisy_latents = flow_scheduler.add_noise(
        clean_latent.flatten(0, 1),
        noise.flatten(0, 1),
        timestep.flatten(0, 1)
    ).unflatten(0, (batch_size, num_frame))
    
    if model.noise_augmentation_max_timestep > 0:
        index_clean_aug = model._get_timestep(
            0,
            model.noise_augmentation_max_timestep,
            clean_latent.shape[0],
            clean_latent.shape[1],
            model.num_frame_per_block,
            uniform_timestep=False
        )
        timestep_clean_aug = flow_scheduler.timesteps[index_clean_aug].to(dtype=model.dtype, device=model.device)
        clean_latent_aug =flow_scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep_clean_aug.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
    else:
        clean_latent_aug = clean_latent
        timestep_clean_aug = None
    
    output_tensor_list = model(
            noisy_latents=noisy_latents, 
            timestep=timestep, 
            conditional_dict=prompt_emb["context"],
            unconditional_dict=prompt_emb["unconditional_dict"],
            clean_latent_aug=clean_latent_aug,
            timestep_clean_aug=timestep_clean_aug
        )
    
    loss = torch.nn.functional.mse_loss(
        output_tensor_list.float(), training_target.float()
    )
    loss_wo_w = loss
    loss = loss * flow_scheduler.flow_scheduler.training_weight(timestep)
    # print("loss", loss)
    return [loss, loss_wo_w], loss_func

if __name__ == "__main__":
    # wait_for_debugger(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # main()
    args = parse_args(extra_args=extra_args)
    args.distributed_vae = None
    trainer = CausalTrainer(args)
    trainer.pretrain(forward_step_func=forward_step)
