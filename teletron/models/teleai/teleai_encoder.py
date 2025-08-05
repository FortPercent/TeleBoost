
import torch
from typing import Dict, Any, Tuple, List,Union
from teletron.core.distributed.base_encoder import BaseEncoder
from teletron.models.teleai.models.dit.teleai_dit import TeleaiPrompter
from teletron.models.teleai.pipelines.teleai.teleai_video import TeleaiVideoPipeline
from teletron.models.teleai.models.dit.teleai_dit import ModelManager
from teletron.models.teleai.teleai_encoder_utils import (
    get_context,
    get_img_clip_feature,
    get_img_emb_y,
    get_latents,
    get_noise,
    get_fake_latents
)
from teletron.utils import get_args
from functools import partial

ENCODER_SCHEMA = {
    'teleai_i2v': ['context', 'img_clip_feature', 'img_emb_y', 'latents'],
    # 'teleai_moe': ['context', 'img_clip_feature', 'img_emb_y', 'latents', 'noise'],
    'teleai_sr': ['context', 'img_clip_feature', 'img_emb_y', 'latents', 'fake_latents'],
    'teleai_multimask': ['context', 'img_clip_feature', 'img_emb_y', 'latents'],
}

WORK_FN = {
    'context': get_context,
    'img_clip_feature': get_img_clip_feature,
    'img_emb_y': get_img_emb_y,
    'latents': get_latents,
    'noise': get_noise,
    'fake_latents': get_fake_latents,
}

PROPERTY_DIMS = {
    'context': 3,
    'img_clip_feature': 3,
    'img_emb_y': 5,
    'latents': 5,
    'nosie': 5,
    'fake_latents': 5,
}


class TeleaiEncoder(BaseEncoder):
    """Teleai视频模型的具体编码器实现。"""

    @staticmethod
    def get_output_schema() -> List[str]:
        """返回此编码器输出张量的固定名称和顺序。"""
        args = get_args()
        return ENCODER_SCHEMA[args.task_type]

    def __init__(self, device: torch.device, **kwargs: Any):
        super().__init__(device)
        args = get_args()
        kwargs['model_paths'] = args.encoder_model_path
        kwargs['tokenizer_path'] = args.encoder_tokenizer_path

        kwargs['tiler_kwargs'] = {
            "tiled": True, 
            "tile_size":  (34, 34), 
            "tile_stride": (18, 16)
        }
        self.model_paths = kwargs.get("model_paths")
        self.tokenizer_path = kwargs.get("tokenizer_path")
        self.tiler_kwargs = kwargs.get("tiler_kwargs", {})

        if not self.model_paths or not self.tokenizer_path:
            raise ValueError("TeleaiEncoder需要 'model_paths' 和 'tokenizer_path' 参数。")

        # 将模型组件初始化为None，它们将在setup()中被加载
        self.text_encoder = None
        self.image_encoder = None
        self.vae = None
        self.prompter = None
        self.work_fn = WORK_FN

    def setup(self) -> None:
        """加载所有必需的teleai模型组件到指定设备。"""
        print(f"在设备 {self.device} 上设置 TeleaiEncoder...")
        
        model_manager = ModelManager(torch_dtype=torch.float32, device="cpu")
        model_manager.load_models(self.model_paths)
        
        pipe = TeleaiVideoPipeline.from_model_manager(model_manager)
        
        self.text_encoder = pipe.text_encoder.to(device=self.device, dtype=torch.bfloat16)
        self.image_encoder = pipe.image_encoder.to(device=self.device)
        self.vae = pipe.vae.to(device=self.device, dtype=torch.bfloat16)
        del pipe # 释放不再需要的内存

        self.prompter = TeleaiPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(self.tokenizer_path)

        for key, val in self.work_fn.items():
            self.work_fn[key] = self.prepare_work_fn(key, val)

        print("TeleaiEncoder 设置完成。")

    def prepare_work_fn(self, target, work_fn):
        if target == 'context':
            return partial(work_fn, prompter=self.prompter, dtype=torch.bfloat16)
        elif target == 'img_clip_feature':
            return partial(work_fn, image_encoder=self.image_encoder, dtype=torch.bfloat16)
        elif target == 'img_emb_y':
            return partial(work_fn, vae=self.vae, dtype=torch.bfloat16)
        elif target == 'latents':
            return partial(work_fn, vae=self.vae, dtype=torch.bfloat16)
        elif target == 'noise':
            return partial(work_fn, dtype=torch.bfloat16)
        elif target == 'fake_latents':
            return partial(work_fn, vae=self.vae, dtype=torch.bfloat16)
        else:
            return work_fn
    
    def encode(self, raw_batch: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[List[Any], List[List[Any]]]:
        """
        使用teleai模型对数据批次进行编码。
        
        Args:
            raw_batch: 单个数据样本（字典）或一批数据样本（字典列表）。

        Returns:
            如果输入是单个样本，返回编码后的张量列表。
            如果输入是样本列表，返回一个包含两个列表的列表，分别对应每个样本的编码结果。
        """
        batch = {}
        for data_to_produce in self.get_output_schema():

            batch[data_to_produce] = self.work_fn[data_to_produce](batch=raw_batch)

        if isinstance(raw_batch, list):
            schema = self.get_output_schema()
            batch0 = [batch[key][0] for key in schema]
            batch1 = [batch[key][1] for key in schema]
            return [batch0, batch1]
        else:
            return [batch[key] for key in self.get_output_schema()]