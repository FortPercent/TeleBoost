import torch
import torch.distributed as dist
from megatron.core import mpu

from teleboost.models.flow_match import FlowMatchScheduler
from teleboost.train import Trainer, parse_args
from teleboost.train.utils import average_losses_across_data_parallel_group

def wan_loss_func(output_tensor):
    """
    output_tensor: [loss]
    """
    loss = output_tensor[0].mean()

    averaged_loss = average_losses_across_data_parallel_group([loss])

    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}



def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    # group.add_argument("--test_valid", type=str, default="")
    group.add_argument("--moe-step-factor-list", type=float, action='append')
    group = parser.add_argument_group(title='encoder args')
    group.add_argument("--encoder-model-path", type=str, default=None)
    group.add_argument("--encoder-tokenizer-path", type=str, default=None)
    return parser

def forward_step(data_iterator, model):
    """
    data_iterator: iterator yielding dataset batches (dict)
    model: WanTrainingModule (wrapped by DDP / FSDP / etc.)
    """
    # 1. 取一个 batch
    batch = next(data_iterator)

    # 2. 前向：WanTrainingModule 内部返回 loss
    #    - SFT: 返回 scalar loss
    #    - DPO: 返回 dpo_loss
    loss = model(batch)

    # 3. 保持 Megatron 期望的返回格式
    return [loss], wan_loss_func



if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)