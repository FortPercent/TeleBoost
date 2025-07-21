import argparse
import os
from omegaconf import OmegaConf

from teletron.train import DiffusionTrainer, parse_args


def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    group.add_argument("--no_save", action="store_false")
    group.add_argument("--load_raw_video", action="store_false")
    group.add_argument("--gradient-checkpointing", action="store_false")
    group.add_argument("--real-name", type=str, default="Wan2.1-T2V-14B")
    group.add_argument("--negative_prompt",type=str,
                       default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    group.add_argument("--base_paths", typr=str, nargs = '+',
                       default=['/nvfile-heatstorage/teleai-infra/kaikai/HumanData_subset_500/merged_videos_latents',]
                       )
    group.add_argument("--metadata_paths", typr=str, nargs = '+',
                       default=['/nvfile-heatstorage/teleai-infra/kaikai/HumanData_subset_500/filtered_500.csv',]
                       )
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    # parser.add_argument("--save", action="store_true")
    parser.add_argument("--save", type=str, default="wan_experiments_test", help="Path to the directory to save logs")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("/nvfile-heatstorage/teleai-infra/kaikai/dreamingforcing/WorldVideo/configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    trainer = DiffusionTrainer(config)
    # breakpoint()
    trainer.train()



if __name__ == "__main__":
    main()
    # args = parse_args(extra_args=extra_args)
    # trainer = DiffusionTrainer(args)
    # trainer.train()
