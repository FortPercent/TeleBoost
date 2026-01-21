
import os
import torch
from typing import Dict, Any, Tuple, List,Union

from teletron.core.distributed.base_encoder import BaseEncoder
from teletron.models.teleai.models.dit import TeleaiPrompter
from teletron.models.teleai.models.dit import TeleaiTextEncoder
from teletron.models.teleai.models.dit import TeleaiVideoVAE
from teletron.models.teleai.models.dit import TeleaiVideoVAE_2_2
from teletron.models.teleai.models.dit import TeleaiImageEncoder
from teletron.models.teleai.models.dit.diffsynth_wan_video_vae import (
    WanVideoVAE as DiffSynthWanVideoVAE,
    WanVideoVAE38 as DiffSynthWanVideoVAE38,
)
from teletron.models.teleai.teleai_encoder_utils import (
    get_context,
    get_img_clip_feature,
    get_img_emb_y,
    get_latents,
    get_noise,
    get_fake_latents,
    get_unprompt_emb,
    get_depth_latents,
)
from teletron.utils import get_args, set_config

from functools import partial

WORK_FN = {
    'context': get_context,
    'img_clip_feature': get_img_clip_feature,
    'img_emb_y': get_img_emb_y,
    'latents': get_latents,
    'noise': get_noise,
    'fake_latents': get_fake_latents,
    'prompt_emb': get_context,
    'unprompt_emb': get_unprompt_emb,
    'depth_latents': get_depth_latents,
}

PROPERTY_DIMS = {
    'context': 3,
    'img_clip_feature': 3,
    'img_emb_y': 5,
    'latents': 5,
    'noise': 5,
    'fake_latents': 5,
    'prompt_emb': 3,
    'unprompt_emb': 3,
    'depth_latents': 5,
}


def _dump_model_state_dict(model, model_name, torch_dtype):
    out_dir = os.path.join(os.getcwd(), "model_dumps")
    os.makedirs(out_dir, exist_ok=True)
    dtype_label = str(torch_dtype).replace("torch.", "")
    out_path = os.path.join(out_dir, f"teletron_{model_name}_{dtype_label}.pth")
    try:
        torch.save(model.state_dict(), out_path)
        print(f"[TeleaiEncoder] saved model state_dict to {out_path}")
    except Exception as exc:
        print(f"[TeleaiEncoder] failed to save model state_dict: {exc}")


class TeleaiEncoder(BaseEncoder):
    """Teleai视频模型的具体编码器实现。"""

    @staticmethod
    def get_output_schema() -> List[str]:
        """返回此编码器输出张量的固定名称和顺序。"""
        return set_config().get("model_config", None).get("encoder", None).get("encoder_schema", ['context', 'latents'])

    def __init__(self, device: torch.device):
        super().__init__(device)
        encoder_model_config = set_config().get("model_config", None).get("encoder", None)
        if encoder_model_config is None:
            raise ValueError("未找到encoder模型配置。")

        self.vae_path = encoder_model_config.get("vae", None).get("path", None)
        self.vae_type = encoder_model_config.get("vae", None).get("type", "TeleaiVideoVAE_2_1")
        self.tiler_kwargs = encoder_model_config.get("vae", None).get("tiler_kwargs", {})
        self.vae_compile = encoder_model_config.get("vae", None).get("torch_compile", False)
        self.compression_cfg = encoder_model_config.get("vae", None).get("compression", None)
        if self.tiler_kwargs is None:
            self.tiler_kwargs = dict(
                tiled=False,
                tile_size=(34, 34),
                tile_stride=(18, 16),
            )
        self.text_encoder_path = encoder_model_config.get("text_encoder", None).get("path", None)
        self.tokenizer_path = encoder_model_config.get("text_encoder", None).get("tokenizer_path", None)
        
        if encoder_model_config.get("image_encoder", None) is not None:
            self.image_encoder_path = encoder_model_config.get("image_encoder", None).get("path", None)
        else:
            self.image_encoder_path = None
        if encoder_model_config.get("image_encoder", None) is not None:
            self.image_encoder_compile = encoder_model_config.get("image_encoder", None).get("torch_compile", False)
        else:
            self.image_encoder_compile = None
        
        if encoder_model_config.get("depth_model", None) is not None:
            self.depth_model_path = encoder_model_config.get("depth_model", None).get("path", None)
        else:
            self.depth_model_path = None

        if not self.vae_path or not self.text_encoder_path or not self.tokenizer_path:
            raise ValueError("TeleaiEncoder需要 'text_encoder_path' 和 'tokenizer_path' 参数。")

        # 将模型组件初始化为None，它们将在setup()中被加载
        self.text_encoder = None
        self.image_encoder = None
        self.vae = None
        self.prompter = None
        self.depth_model = None
        self.work_fn = WORK_FN

    def setup(self) -> None:
        """加载所有必需的teleai模型组件到指定设备。"""
        print(f"在设备 {self.device} 上设置 TeleaiEncoder...")
        print(
            "Init VAE params: "
            f"type={self.vae_type} path={self.vae_path} "
            f"tiler_kwargs={self.tiler_kwargs} torch_compile={self.vae_compile}"
        )
        print(f"加载 VAE 模型... {self.vae_path}")
        if self.vae_type in ("DiffSynthWanVideoVAE", "diffsynth_wan_video_vae"):
            self.vae = DiffSynthWanVideoVAE().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            if self.vae_compile and hasattr(self.vae, "model") and hasattr(self.vae.model, "encode"):
                self.vae.model.encode = torch.compile(self.vae.model.encode, dynamic=True)
                print("torch.compile DiffSynth VAE model... ")
            if self.compression_cfg is not None:
                self.compression = tuple(self.compression_cfg)
            else:
                self.compression = (4, 8, 8)
        elif self.vae_type in ("DiffSynthWanVideoVAE38", "diffsynth_wan_video_vae38"):
            self.vae = DiffSynthWanVideoVAE38().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            if self.vae_compile and hasattr(self.vae, "model") and hasattr(self.vae.model, "encode"):
                self.vae.model.encode = torch.compile(self.vae.model.encode, dynamic=True)
                print("torch.compile DiffSynth VAE38 model... ")
            if self.compression_cfg is not None:
                self.compression = tuple(self.compression_cfg)
            else:
                self.compression = (4, 8, 8)
        elif self.vae_type == "TeleaiVideoVAE_2_1":
            self.vae = TeleaiVideoVAE().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            if self.vae_compile:
                self.vae.model.encode = torch.compile(self.vae.model.encode, dynamic=True)
                print(f"torch.compile VAE 模型... ")
            if self.compression_cfg is not None:
                self.compression = tuple(self.compression_cfg)
            else:
                self.compression = (4, 8, 8)
        else:
            self.vae = TeleaiVideoVAE_2_2().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            if self.compression_cfg is not None:
                self.compression = tuple(self.compression_cfg)
            else:
                self.compression = (4, 16, 16)
        self.vae.model.load_state_dict(torch.load(self.vae_path, map_location='cpu', weights_only=False), strict=True)
        _dump_model_state_dict(self.vae.model, self.vae_type, torch.bfloat16)

        print(f"加载 Text Encoder 模型... {self.text_encoder_path}")
        self.text_encoder = TeleaiTextEncoder().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(torch.load(self.text_encoder_path, map_location='cpu', weights_only=False), strict=True)
        self.prompter = TeleaiPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(self.tokenizer_path)

        if self.image_encoder_path is not None:
            print(f"加载 Image Encoder 模型... {self.image_encoder_path}")
            self.image_encoder = TeleaiImageEncoder().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.image_encoder.model.load_state_dict(torch.load(self.image_encoder_path, map_location='cpu', weights_only=False), strict=False)

        if self.depth_model_path is not None:
            print(f"加载 Depth Model 模型... {self.depth_model_path}")
            from video_depth_anything.video_depth import VideoDepthAnything
            self.depth_model = VideoDepthAnything().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.depth_model.load_state_dict(torch.load(self.depth_model_path, map_location='cpu', weights_only=False), strict=True)

        if self.image_encoder_compile:
            self.image_encoder.encode_image = torch.compile(self.image_encoder.encode_image)
            print(f"torch.compile Image Encoder 模型... ")
        for key, val in self.work_fn.items():
            self.work_fn[key] = self.prepare_work_fn(key, val)

        print("TeleaiEncoder 设置完成。")

    def prepare_work_fn(self, target, work_fn):
        if target == 'context':
            return partial(work_fn, prompter=self.prompter, dtype=torch.bfloat16)
        elif target == 'img_clip_feature':
            return partial(work_fn, image_encoder=self.image_encoder, dtype=torch.bfloat16)
        elif target == 'img_emb_y':
            return partial(work_fn, vae=self.vae, dtype=torch.bfloat16, compression=self.compression, tiler_kwargs=self.tiler_kwargs)
        elif target == 'latents':
            return partial(work_fn, vae=self.vae, dtype=torch.bfloat16, tiler_kwargs=self.tiler_kwargs)
        elif target == 'noise':
            return partial(work_fn, dtype=torch.bfloat16, compression=self.compression)
        elif target == 'fake_latents':
            return partial(work_fn, vae=self.vae, dtype=torch.bfloat16, tiler_kwargs=self.tiler_kwargs)
        elif target == 'prompt_emb':
            return partial(work_fn, prompter=self.prompter, dtype=torch.bfloat16)
        elif target == 'unprompt_emb':
            if not getattr(self, "unprompt_emb", None):
                self.unprompt_emb = partial(work_fn, prompter=self.prompter, dtype=torch.bfloat16)
            return self.unprompt_emb
        elif target == 'depth_latents':
            return partial(work_fn, depth_model=self.depth_model, vae=self.vae, dtype=torch.bfloat16, tiler_kwargs=self.tiler_kwargs)
        else:
            return work_fn
    
    def _is_dpo_batch(self, raw_batch: dict) -> bool:

        dataset_config =set_config().get("dataset",{})
        chosen_key = dataset_config.get("chosen_video_key","chosen")
        rejected_key = dataset_config.get("rejected_video_key","rejected")
        return (
            isinstance(raw_batch, dict)
            and chosen_key in raw_batch
            and rejected_key in raw_batch
        )

    def _encode_single(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        schema = self.get_output_schema()
        out = {}
        for key in schema:
            out[key] = self.work_fn[key](batch=raw_batch)
        return out

    def _encode_dpo(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_config = set_config().get("dataset", {})
        chosen_key = dataset_config.get("chosen_video_key", "chosen")
        rejected_key = dataset_config.get("rejected_video_key", "rejected")
        branches = (chosen_key, rejected_key)

        schema = self.get_output_schema()
        out = {}

        # ========== 1️⃣ prompt / context：只算一次 ==========
        shared_input = raw_batch[chosen_key]  # chosen / rejected 共用 prompt

        for key in schema:
            if key in ["context", "prompt_emb", "unprompt_emb"]:
                out[key] = self.work_fn[key](batch=shared_input)

        # ========== 2️⃣ image / latent：CUDA stream 并行 ==========
        streams = {
            chosen_key: torch.cuda.Stream(device=self.device),
            rejected_key: torch.cuda.Stream(device=self.device),
        }

        branch_outputs = {}

        for branch in branches:
            branch_outputs[branch] = {}
            branch_input = raw_batch[branch]

            for key in schema:
                if key in ["context", "prompt_emb", "unprompt_emb"]:
                    continue

                with torch.cuda.stream(streams[branch]):
                    branch_outputs[branch][key] = self.work_fn[key](batch=branch_input)

        # ========== 3️⃣ 同步 ==========
        for s in streams.values():
            s.synchronize()

        out.update(branch_outputs)
        return out


    def encode(self, raw_batch: Union[Dict[str, Any]]) -> Union[List[Any], List[List[Any]]]:
        """
        使用teleai模型对数据批次进行编码。
        
        Args:
            raw_batch: 单个数据样本（字典）或一批数据样本（字典列表）。

        Returns:
            如果输入是单个样本，返回编码后的张量列表。
            如果输入是样本列表，返回一个包含两个列表的列表，分别对应每个样本的编码结果。
        """
        if not self._is_dpo_batch(raw_batch):
            return self._encode_single(raw_batch)
        else:
            return self._encode_dpo(raw_batch)
                
