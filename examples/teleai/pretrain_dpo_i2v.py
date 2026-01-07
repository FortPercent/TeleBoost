import torch
from teletron.train import Trainer, parse_args
import torch.distributed as dist
from megatron.core import mpu
from teletron.models.flow_match import FlowMatchScheduler
from teletron.train.utils import get_batch, loss_func as base_loss_func, average_losses_across_data_parallel_group
from teletron.utils import get_timers, set_config
import torch.nn.functional as F

def dpo_loss_func(output_tensor):
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}


def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    # group.add_argument("--test_valid", type=str, default="")
    group.add_argument("--moe-step-factor-list", type=float, action='append')
    group.add_argument("--test-with-pseudo-data", action="store_true")
    group.add_argument("--test-resolution", type=str, default="360")
    
    return parser

def _compute_single_loss(
    latents,
    context,
    clip_feature,
    y,
    flow_scheduler,
    model,
    timestep,
    noise,
):
    training_target = flow_scheduler.training_target(latents, noise, timestep)
    noisy_latents = flow_scheduler.add_noise(latents, noise, timestep)
    loss_weight = flow_scheduler.training_weight(timestep)

    if args.test_with_pseudo_data:
        dp_rank = mpu.get_data_parallel_rank() % 2
        curr_iter = args.curr_iteration % 1000
        input_dict = torch.load(
            f"../test_data/saved_inputs_{args.test_resolution}/input_dict_iter{curr_iter}_rank{dp_rank}.pt",
            weights_only=False,
            map_location="cpu",
        )
        output_tensor_list = model(
            x=input_dict["noisy_latents"].cuda(),
            timestep=input_dict["timestep"].cuda(),
            context=input_dict["prompt_emb"]["context"].cuda(),
            clip_feature=input_dict["image_emb"]["clip_feature"].cuda(),
            y=input_dict["image_emb"]["y"].cuda(),
        )
        training_target = input_dict["training_target"].cuda()
        loss_weight = input_dict["loss_weight"].cuda()
    else:
        output_tensor_list = model(
            x=noisy_latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
        )
    print(f"noisy_latents = {noisy_latents.shape},output_tensor_list = {output_tensor_list.shape} training_target = {training_target.shape}")
    loss = torch.nn.functional.mse_loss(
        output_tensor_list.float(), training_target.float()
    )
    loss_wo_w = loss
    loss = loss * loss_weight

    first_frame_pred = output_tensor_list[:, :, :1, :, :]
    first_frame_target = training_target[:, :, :1, :, :]
    assert first_frame_pred.shape[1] == 16
    first_frame_loss = torch.nn.functional.mse_loss(
        first_frame_pred.float(), first_frame_target.float()
    )
    loss += first_frame_loss

    return loss, loss_wo_w, first_frame_loss


def forward_step(data_iterator, model, time_step=None):
    flow_scheduler = FlowMatchScheduler(shift=1, sigma_min=0.0, extra_one_step=True)
    flow_scheduler.set_timesteps(1000, training=True)

    timers = get_timers()
    timers.start_timer('get-data-time')
    batch = next(data_iterator)
    timers.stop_timer('get-data-time')
    context = batch["context"]  # shared text context

    chosen_latents = batch["chosen"]["latents"]
    reject_latents = batch["rejected"]["latents"]

    # clip feature / img_emb_y（正负样本各自的）
    chosen_clip_feature = batch["chosen"].get(
        "img_clip_feature", batch["chosen"].get("clip_feature")
    )
    reject_clip_feature = batch["rejected"].get(
        "img_clip_feature", batch["rejected"].get("clip_feature")
    )

    chosen_y = batch["chosen"].get("img_emb_y")
    reject_y = batch["rejected"].get("img_emb_y")

    # =========================
    # timestep sampling
    # =========================
    diffusion_config = (
        set_config()
        .get("model_config", {})
        .get("training", {})
        .get("diffusion", {})
    )

    min_timestep_boundary = int(
        diffusion_config.get("min_timestep_boundary")
        * flow_scheduler.num_train_timesteps
    )
    max_timestep_boundary = int(
        diffusion_config.get("max_timestep_boundary")
        * flow_scheduler.num_train_timesteps
    )

    timestep_range = [min_timestep_boundary, max_timestep_boundary]
    timestep_id = torch.randint(
        timestep_range[0], timestep_range[1], (1,)
    )
    timestep = flow_scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
    )

    def broadcast_tensor(input: torch.Tensor):
        tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
        if mpu.get_tensor_context_parallel_world_size() > 1:
            dist.broadcast(
                input,
                tp_cp_src_rank,
                group=mpu.get_tensor_context_parallel_group(),
            )

    if time_step is not None:
        timestep = torch.tensor(
            [time_step],
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )

    broadcast_tensor(timestep)

    # =========================
    # noise (正负样本独立)
    # =========================
    noise_chosen = torch.randn_like(chosen_latents)
    noise_reject = torch.randn_like(reject_latents)

    broadcast_tensor(noise_chosen)
    broadcast_tensor(noise_reject)

    # =========================
    # forward & loss
    # =========================
    loss_chosen, loss_wo_w_chosen, first_frame_loss_chosen = (
        _compute_single_loss(
            chosen_latents,
            context,
            chosen_clip_feature,
            chosen_y,
            flow_scheduler,
            model,
            timestep,
            noise_chosen,
        )
    )

    loss_reject, loss_wo_w_reject, first_frame_loss_reject = (
        _compute_single_loss(
            reject_latents,
            context,
            reject_clip_feature,
            reject_y,
            flow_scheduler,
            model,
            timestep,
            noise_reject,
        )
    )

    beta = float(
        set_config()
        .get("model_config")
        .get("dit")
        .get("train")
        .get("dpo")
        .get("beta")
    )

    advantage = (loss_reject - loss_chosen).clamp(-20, 20)
    dpo_loss = -F.logsigmoid(beta * advantage).mean()

    return [dpo_loss], dpo_loss_func


if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)
