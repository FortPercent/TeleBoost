from teletron.utils import set_config
import torch
import torch.nn.functional as F
from .diffsynth_diffusion_training_module import DiffusionTrainingModule
from .pipeline.wan_video_pipeline import WanVideoPipeline

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        config,
        model_paths=None, model_id_with_origin_paths=None,
        reference_model_paths=None,
        reference_model_id_with_origin_paths=None,
    ):
        super().__init__()

        cfg = set_config()
        model_cfg = cfg.get("model_config", {})
        dit_cfg = model_cfg.get("dit", {})
        train_cfg = dit_cfg.get("train", {})
        training_cfg = model_cfg.get("training", {})

        self.trainable_models = train_cfg.get("trainable_models", "dit")

        self.use_gradient_checkpointing = train_cfg.get("use_gradient_checkpointing", True)
        self.use_gradient_checkpointing_offload = train_cfg.get("use_gradient_checkpointing_offload", False)

        lora_cfg = train_cfg.get("lora", {})
        self.lora_enable = lora_cfg.get("enable", False)
        self.lora_base_model = lora_cfg.get("base_model", None)
        self.lora_target_modules = lora_cfg.get("target_modules", "q,k,v,o,ffn.0,ffn.2")
        self.lora_rank = lora_cfg.get("rank", 32)
        self.lora_checkpoint = lora_cfg.get("checkpoint", None)

        dpo_cfg = train_cfg.get("dpo", {})
        self.dpo = dpo_cfg.get("enable", False)
        self.dpo_beta = dpo_cfg.get("beta", 0.1)

        self.extra_inputs = train_cfg.get("extra_inputs", [])

        diff_cfg = training_cfg.get("diffusion", {})
        self.max_timestep_boundary = diff_cfg.get("max_timestep_boundary", 1.0)
        self.min_timestep_boundary = diff_cfg.get("min_timestep_boundary", 0.0)

        io_cfg = training_cfg.get("dpo_io", {})
        self.chosen_key = io_cfg.get("chosen_key", "chosen")
        self.rejected_key = io_cfg.get("rejected_key", "rejected")

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            enable_fp8_training=False,
        )

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
        )

        if getattr(self.pipe, "vae", None) is not None:
            vae = self.pipe.vae
            self.pipe._modules.pop("vae", None)
            object.__setattr__(self.pipe, "vae", vae)
            vae.requires_grad_(False)

        self.switch_pipe_to_training_mode(
            pipe=self.pipe,
            trainable_models=self.trainable_models,
            lora_base_model=self.lora_base_model if self.lora_enable else None,
            lora_target_modules=self.lora_target_modules,
            lora_rank=self.lora_rank,
            lora_checkpoint=self.lora_checkpoint,
            enable_fp8_training=False,
        )


    def prepare_extra_modules(self, accelerator):
        """
        Move auxiliary modules to the correct device after `accelerator.prepare`.
        """
        self.pipe.vae.to(accelerator.device)
        self.pipe.device = accelerator.device
    
    def forward_preprocess(self, data, video_key="video"):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data[video_key],
            "height": data[video_key][0].size[1],
            "width": data[video_key][0].size[0],
            "num_frames": len(data[video_key]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        pair_id = data.get("dpo_pair_id")
        if pair_id is not None:
            inputs_shared["dpo_pair_id"] = pair_id
        inputs_shared["dpo_branch"] = data.get("dpo_branch", video_key)
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data[video_key][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data[video_key][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if not self.dpo:
            if inputs is None:
                inputs = self.forward_preprocess(data)
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            return self.pipe.training_loss(**models, **inputs)

        chosen_inputs = self.forward_preprocess({
            "prompt": data["prompt"],
            self.chosen_key: data[self.chosen_key],
        }, video_key=self.chosen_key)
        rejected_inputs = self.forward_preprocess({
            "prompt": data["prompt"],
            self.rejected_key: data[self.rejected_key],
        }, video_key=self.rejected_key)

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}

        b = chosen_inputs["input_latents"].shape[0]
        idx = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (b,), device=self.pipe.scheduler.timesteps.device)
        timesteps = self.pipe.scheduler.timesteps[idx].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        loss_chosen = self.pipe.training_loss(**models, **chosen_inputs, timesteps=timesteps, reduction='none')
        loss_rejected = self.pipe.training_loss(**models, **rejected_inputs, timesteps=timesteps, reduction='none')

        advantage = (loss_rejected - loss_chosen).clamp(-20, 20)
        return -F.logsigmoid(self.dpo_beta * advantage).mean()

