# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
import json
import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu

from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0,get_model
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.training import pretrain
from vast.train.configs.config import load_config
from megatron.training.utils import (
    average_losses_across_data_parallel_group
)
import torch.distributed as dist
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import (
    get_args,
    get_timers,
)

from teletron.models.wan.pipeline import WanPipeline
from teletron.training.utils import get_batch_on_this_tp_cp_rank_vast
from teletron.datasets.utils import train_valid_test_datasets_provider, load_config_vast

# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.beFFnchmark = False

# batch = {
#     "images": torch.randn(1, 9, 3, 240, 128).to(torch.float32).to(device='cuda'),
#     "first_ref_image": torch.randn(1, 1, 3, 240, 128).to(torch.float32).to(device='cuda'),
#     "prompt_embeds": torch.randn(1, 256, 4096).to(torch.float32).to(device='cuda'),
#     "clip_text_embed": torch.randn(1, 768).to(torch.float32).to(device='cuda'),
#     "prompt_masks": torch.randint(0, 2, (1, 256), dtype=torch.int64).to(device='cuda'),
# }

class Config(dict):
    def __init__(self, d=None):
        if d is None:
            d = {}
        super().__init__(d)
        for k, v in d.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)


def get_batch(data_iterator):
    # get batches based on the TP_CP rank you are on
    batch = get_batch_on_this_tp_cp_rank_vast(data_iterator)
    # batch = get_batch_on_this_tp_rank_vast(data_iterator)
    return batch


def extra_args_provider(parser):
    group = parser.add_argument_group(title='dataset')
    group.add_argument('--dataset-type', default="KoalaDataset")
    group.add_argument("--num-frames", type=int, default=9,
                       help='number of frames to train, must be of 4n+1, \
                        overloads yaml if using koala dataset. example: 45')
    group.add_argument("--video-resolution", nargs=2, type=int, default=[1280, 720], 
                       help='video resolution to train, overloads yaml if using koala dataset. \
                       width and height should satisfy: (width or height) // 8 % 2 == 0')
    group.add_argument("--koala-opt", type=str, default="./teletron/datasets/koala.yml", 
                        help="the koala dataset option file")


    group = parser.add_argument_group(title="diffusion")
    group.add_argument("--vae-slicing", action="store_false")
    group.add_argument("--vae-tiling", action="store_false")
    group.add_argument("--flow-resolution-shifting", action="store_true")
    group.add_argument("--flow-base-image-seq-len", type=int, default=256)
    group.add_argument("--flow-max-image-seg-len", type=int, default=4096)
    group.add_argument("--flow-base-shift", type=float, default=0.5)
    group.add_argument("--flow-max-shift", type=float, default=1.15)
    group.add_argument("--flow-shift", type=float, default=1.0)
    group.add_argument("--flow-weighting-scheme", type=str, default="none")
    group.add_argument("--flow-logit-mean", type=float, default=0.0)
    group.add_argument("--flow-logit-std", type=float, default=1.0)
    group.add_argument("--flow-mode-scale", type=float, default=1.29)
    
    group = parser.add_argument_group(title='debug')
    group.add_argument("--debug", action="store_true")
    group.add_argument("--debug_dir", type=str, default="./logs")
    group.add_argument("--sanity-check", action="store_true")

    group = parser.add_argument_group(title='training')
    group.add_argument("--task-type", type=str, choices=['wan_flf'], default="wan_flf")
    return parser

# def init(
#     train_valid_test_dataset_provider,
#     model_provider,
#     model_type,
#     forward_step_func,
#     process_non_loss_data_func=None,
#     extra_args_provider=None,
#     args_defaults={},
#     get_embedding_ranks=None,
#     get_position_embedding_ranks=None,
#     non_loss_data_func=None,
# ):
#     initialize_megatron(
#         extra_args_provider=extra_args_provider,
#         args_defaults=args_defaults,
#         get_embedding_ranks=get_embedding_ranks,
#         get_position_embedding_ranks=get_position_embedding_ranks
#     )

#     args = get_args()
    

#/workspace/Wan2___1-I2V-14B-480P/google/umt5-xxl

def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, parallel_output=True
) -> WanPipeline:
    args = get_args()
    config = core_transformer_config_from_args(args)
    config_vast = load_config_vast()
    model = WanPipeline(
        wan_config=config_vast.models,
        config=config,
    )
    # model.module = model.modules
    return model


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}

def forward_step(data_iterator, model: WanPipeline):
    """Forward training step.

    Args:
        data_iterator: Iterable dataset.
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    batch = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor_list = model(batch)

    return output_tensor_list, loss_func

if __name__ == "__main__":
    # global_config = Config(config)


    # initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    # args = get_args()
    # args.iteration = 0
    # train_data_iterator, valid_data_iterator, test_data_iterator \
    #         = build_train_valid_test_data_iterators(
    #             train_valid_test_datasets_provider)
    # batch = get_batch(train_data_iterator)
    # print(batch)
    # checkpoint = torch.load('/nvfile-heatstorage/teleai-infra/adk/Megatron_VAST/ckpt/release/mp_rank_00/model_optim_rng.pt')
    # if 'state_dict' in checkpoint:
    #     model_state_dict = checkpoint['state_dict']
    # else:
    #     model_state_dict = checkpoint
    # model = model_provider()
    # model.load_state_dict(model_state_dict['model'], strict=False)
    # model = model.to(torch.bfloat16).to(device='cuda')
    # output = model(batch)

    # print(batch)
    # print(output)
    
    # from vast.datasets.datasets.build import build_dataset
    # train_ds_config = global_config.dataloaders.train

    # print_rank_0("> building train, validation, and test datasets for multimodal ...")
    
    # dataset = build_dataset(train_ds_config.dataset)
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=extra_args_provider,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )


# until cat /ranks/ranks.json | jq -r '.status' | grep -c ready; do
# echo waiting for ranks ready; sleep 2; done
# cat /ranks/ranks.json| jq --arg hostname "$(echo $HOSTNAME | tr '[:upper:]' '[:lower:]')" '.pod_rank[] | select(.node | ascii_downcase == $hostname) | .rank_id'| xargs -I {} echo "export RANK={}" > .env && source .env
# echo "RANK=$RANK"
# export NCCL_DEBUG=INFO && NCCL_ALGO=RING && python3 tools/train.py projects/hunyuanvideo/configs/hunyuanvideo_i2vhy_sp2_720p_85_24fps_0512.py
# cd /nvfile-heatstorage/hyc/vast