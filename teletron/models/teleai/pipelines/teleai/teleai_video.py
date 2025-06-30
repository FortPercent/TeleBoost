import types
from teletron.models.teleai.models.dit.teleai_dit import ModelManager
from teletron.models.teleai.models.dit.teleai_dit import  TeleaiModel
from teletron.models.teleai.models.dit.teleai_dit import TeleaiTextEncoder
from teletron.models.teleai.models.dit.teleai_dit import TeleaiVideoVAE
from teletron.models.teleai.models.dit.teleai_dit import TeleaiImageEncoder
from teletron.models.teleai.models.dit.teleai_dit import TeleaiPrompter


from .base import BasePipeline
import torch, os





class TeleaiVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.prompter = TeleaiPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: TeleaiTextEncoder = None
        self.image_encoder: TeleaiImageEncoder = None
        self.dit: TeleaiModel = None
        self.vae: TeleaiVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder']
        

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("teleai_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("teleai_video_dit")
        self.vae = model_manager.fetch_model("teleai_video_vae")
        self.image_encoder = model_manager.fetch_model("teleai_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None,):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = TeleaiVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        # 预处理并调整图像尺寸
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        
        # 编码图像上下文
        clip_context = self.image_encoder.encode_image([image])
        
        # 创建并处理掩码
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]
        
        # 构建 VAE 输入
        vae_input = torch.concat([
            image.transpose(0, 1),
            torch.zeros(3, num_frames - 1, height, width).to(image.device)
        ], dim=1)
        
        # 使用 VAE 编码
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], 
                            device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=self.torch_dtype, device=self.device)
        
        # 合并掩码和 VAE 输出
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        
        # 确保数据类型和设备一致
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        
        return {"clip_feature": clip_context, "y": y}
    
