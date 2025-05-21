from dataclasses import dataclass

@dataclass
class HunyuanVideoDatasetConfig():
    train_ds_config: dict = None
    eval_ds_config: dict = None