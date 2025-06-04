# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
import json
import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
# from megatron.core.rerun_state_machine import (
#     get_rerun_state_machine,
#     destroy_rerun_state_machine,
#     RerunDataIterator,
#     RerunMode,
# )
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.wan.pipeline import WanPipeline
from megatron.core import mpu
# builder
from hunyuanvideo_dataset_builder import HunyuanVideoDatasetBuilder
# hunyuanvideoconfig
from megatron.core.datasets.hunyuanvideo_dataset_config import HunyuanVideoDatasetConfig

## test
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0,get_model
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.training import pretrain
from vast.train.configs.config import load_config
from wan.configs.wan_flf import config
# from megatron.training.global_vars import set_global_config
# from megatron.training import get_global_config
from megatron.training.utils import (
    get_batch_on_this_tp_cp_rank_vast,
    get_batch_on_this_tp_rank_vast,
    average_losses_across_data_parallel_group
)
import torch.distributed as dist
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import (
    get_args,
    get_timers,
)
import numpy as np
import random

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


def train_valid_test_datasets_provider(train_val_test_num_samples):
    from vast.datasets.datasets.build import build_dataset as build_dataset_vast
    from megatron.core.datasets.fake.build import build_dataset

    global_config = load_config(config)

    train_ds_config = global_config.dataloaders.train
    # eval_ds_config = global_config.dataloaders.eval

    ds_config = HunyuanVideoDatasetConfig(
        train_ds_config=train_ds_config
    )

    print_rank_0("> building train, validation, and test datasets for multimodal ...")
    # train_ds_config.dataset.type=None
    if "FakeDataset" == train_ds_config.dataset.type:
        train_ds = build_dataset(train_ds_config.dataset)
        valid_ds = None
        test_ds = None
    else:
        dataset = build_dataset_vast(train_ds_config.dataset)
        # print_rank_0(f"dataset.shape: {dataset.shape}")
        train_ds, valid_ds, test_ds = HunyuanVideoDatasetBuilder(
            dataset,
            train_val_test_num_samples,
            lambda: True,
            ds_config,
        ).build()

    print_rank_0("> finished creating multimodal datasets ...")

    return train_ds, valid_ds, test_ds

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

    wan_config=global_config.models

    config = core_transformer_config_from_args(args)

    model = WanPipeline(
        wan_config=wan_config,
        config=config,
        tokenizer_path=os.path.join(os.path.dirname(wan_config.get("text_encoder_path")), "google/umt5-xxl")
    )
    list = []
    with open('model_layers.txt', 'w') as f:
        for name in model.transformer.state_dict().keys():
            list.append(name)
            f.write(f"{name}\n")  # 每行写入一个层名称
    print(model.state_dict().keys())
    print("transformer lenth: ", len(list))
    print("vae lenth: ",         len(model.vae.state_dict().keys()))
    print("Image encoder lenth: ", len(model.image_encoder.state_dict().keys()))
    print("transformer lenth: ", len(model.transformer.state_dict().keys()))

    # hooks = register_hooks(model, print_values=True, max_elements=20)
    
    
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
    global_config = load_config(config)
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
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )


# until cat /ranks/ranks.json | jq -r '.status' | grep -c ready; do
# echo waiting for ranks ready; sleep 2; done
# cat /ranks/ranks.json| jq --arg hostname "$(echo $HOSTNAME | tr '[:upper:]' '[:lower:]')" '.pod_rank[] | select(.node | ascii_downcase == $hostname) | .rank_id'| xargs -I {} echo "export RANK={}" > .env && source .env
# echo "RANK=$RANK"
# export NCCL_DEBUG=INFO && NCCL_ALGO=RING && python3 tools/train.py projects/hunyuanvideo/configs/hunyuanvideo_i2vhy_sp2_720p_85_24fps_0512.py
# cd /nvfile-heatstorage/hyc/vast