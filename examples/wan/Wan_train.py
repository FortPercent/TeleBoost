import argparse
import os
from omegaconf import OmegaConf
# import wandb
# import sys
# sys.path.append("..")
from teletron.train import DiffusionTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="wan_experiments_test", help="Path to the directory to save logs")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("/nvfile-heatstorage/teleai-infra/kaikai/dreamingforcing/WorldVideo/configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    trainer.train()



if __name__ == "__main__":
    main()
