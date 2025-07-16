import torch
from teletron.train import Trainer, parse_args
import torch.distributed as dist
from megatron.core import mpu
from teletron.models.flow_match import FlowMatchScheduler
from teletron.train.utils import get_batch, loss_func



def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    # group.add_argument("--test_valid", type=str, default="")
    group.add_argument("--moe-step-factor-list", type=float, action='append')
    group = parser.add_argument_group(title='encoder args')
    group.add_argument("--encoder_model_path", type=str, nargs = '+',default=
                       ['/workspace/Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth', 
                        '/workspace/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth', 
                        '/workspace/Wan2___1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth']
                       )
    group.add_argument("--encoder_tokenizer_path", type=str, default=
                       "/workspace/Wan2___1-I2V-14B-480P/google/umt5-xxl")
    
    return parser

def forward_step(data_iterator, model):
    flow_scheduler = FlowMatchScheduler(shift=1, sigma_min=0.0, extra_one_step=True)
    flow_scheduler.set_timesteps(1000, training=True)
    
    batch = next(data_iterator)
    latents = batch["latents"]
    noise = torch.randn_like(latents) 
    timestep_range = [0, flow_scheduler.num_train_timesteps]

    timestep_id = torch.randint(timestep_range[0], timestep_range[1], (1,))
    
    timestep = flow_scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16, device=torch.cuda.current_device()
    )
    def broadcast_timesteps(input: torch.Tensor):
        tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
        if mpu.get_tensor_context_parallel_world_size() > 1:
            dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())

    broadcast_timesteps(timestep)
    broadcast_timesteps(noise)

    training_target = flow_scheduler.training_target(latents, noise, timestep)    
    noisy_latents = flow_scheduler.add_noise(latents, noise, timestep)
    output_tensor_list = model(x=noisy_latents, 
                               timestep=timestep, 
                               context=batch['context'],
                               clip_feature=batch['img_clip_feature'],
                               y=batch['img_emb_y'])

    loss = torch.nn.functional.mse_loss(
        output_tensor_list.float(), training_target.float()
    )
    loss_wo_w = loss
    loss = loss * flow_scheduler.training_weight(timestep)

    first_frame_pred = output_tensor_list[:, :, :1, :, :]
    first_frame_target = training_target[:, :, :1, :, :]
    assert first_frame_pred.shape[1] == 16
    first_frame_loss = torch.nn.functional.mse_loss(
        first_frame_pred.float(), first_frame_target.float()
    )
    loss += first_frame_loss

    # print("loss", loss)
    return [loss, loss_wo_w, first_frame_loss], loss_func



if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)
