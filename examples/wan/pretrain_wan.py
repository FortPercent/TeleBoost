import torch
from teletron.train import Trainer, parse_args
import torch.distributed as dist
from vast.schedulers.flow_match import FlowMatchScheduler
from megatron.core import mpu
from teletron.train.utils import get_batch, loss_func



def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    # group.add_argument("--test_valid", type=str, default="")
    group = parser.add_argument_group(title='encoder args')
    group.add_argument("--encoder_model_path", type=str, nargs = '+',default=
                       ['/workspace/Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth', 
                        '/workspace/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth', 
                        '/workspace/Wan2___1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pt']
                       )
    group.add_argument("--encoder_tokenizer_path", type=str, default=
                       "/workspace/Wan2___1-I2V-14B-480P/google/umt5-xxl")
    return parser

def forward_step(data_iterator, model):
    flow_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    flow_scheduler.set_timesteps(1000, training=True)
    prompt_emb = {}
    batch = next(data_iterator)
    latents = batch["latents"]
    noise = torch.randn_like(latents) if "noise" not in batch else batch["noise"]

    timestep_id = torch.randint(0, flow_scheduler.num_train_timesteps, (1,))
    timestep = flow_scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16, device=torch.cuda.current_device()
    )
    def broadcast_timesteps(input: torch.Tensor):
        tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
        if mpu.get_tensor_context_parallel_world_size() > 1:
            dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())

    broadcast_timesteps(timestep)
    broadcast_timesteps(noise)
    prompt_emb["context"] = batch["context"]

    image_emb = {}
    image_emb["y"] = batch["image_emb_y"]
    print('y shape', image_emb['y'].shape)
    noisy_latents = flow_scheduler.add_noise(latents, noise, timestep)
    print('x shape', latents.shape)
    image_emb["clip_feature"] = batch["clip_feature"]

    print("batch[clip_feature].shape", batch["clip_feature"].shape)
    print("noisy_latents.shape", noisy_latents.shape)

    output_tensor_list = model(x=noisy_latents, 
                               timestep=timestep, 
                               context=prompt_emb["context"],
                               clip_feature=image_emb["clip_feature"],
                               y=image_emb["y"])

    return output_tensor_list, loss_func



if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)