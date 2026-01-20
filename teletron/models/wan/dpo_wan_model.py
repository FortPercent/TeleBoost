from teletron.utils import set_config
import torch
from .diffsynth_diffusion_training_module import DiffusionTrainingModule
from .pipeline.wan_video_pipeline import WanVideoPipeline

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        config,
        model_paths=None, model_id_with_origin_paths=None,
        # trainable_models=None,
        # lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        # use_gradient_checkpointing=True,
        # use_gradient_checkpointing_offload=False,
        # extra_inputs=None,
        # dpo=False,
        # dpo_beta=0.1,
        # chosen_key="chosen",
        # rejected_key="rejected",
        # max_timestep_boundary=1.0,
        # min_timestep_boundary=0.0,
        # no use
        reference_model_paths=None,
        reference_model_id_with_origin_paths=None,
    ):
        super().__init__()
        # Load models

        # --------------------------------------------------
        # 1. 读取 config（最先）
        # --------------------------------------------------
        cfg = set_config()
        model_cfg = cfg.get("model_config", {})
        dit_cfg = model_cfg.get("dit", {})
        train_cfg = dit_cfg.get("train", {})
        training_cfg = model_cfg.get("training", {})

        # --------------------------------------------------
        # 2. 解析训练语义（⚠️必须在构建 pipe 之前）
        # --------------------------------------------------

        # 哪些子模块可训练
        self.trainable_models = train_cfg.get("trainable_models", "dit")

        # gradient checkpoint
        self.use_gradient_checkpointing = train_cfg.get(
            "use_gradient_checkpointing", True
        )
        self.use_gradient_checkpointing_offload = train_cfg.get(
            "use_gradient_checkpointing_offload", False
        )

        # LoRA
        lora_cfg = train_cfg.get("lora", {})
        self.lora_enable = lora_cfg.get("enable", False)
        self.lora_base_model = lora_cfg.get("base_model", None)
        self.lora_target_modules = lora_cfg.get(
            "target_modules", "q,k,v,o,ffn.0,ffn.2"
        )
        self.lora_rank = lora_cfg.get("rank", 32)
        self.lora_checkpoint = lora_cfg.get("checkpoint", None)

        # DPO
        dpo_cfg = train_cfg.get("dpo", {})
        self.dpo = dpo_cfg.get("enable", False)
        self.dpo_beta = dpo_cfg.get("beta", 0.1)

        # forward contract
        self.extra_inputs = train_cfg.get("extra_inputs", [])

        # diffusion boundary
        diff_cfg = training_cfg.get("diffusion", {})
        self.max_timestep_boundary = diff_cfg.get(
            "max_timestep_boundary", 1.0
        )
        self.min_timestep_boundary = diff_cfg.get(
            "min_timestep_boundary", 0.0
        )

        # DPO IO keys
        io_cfg = training_cfg.get("dpo_io", {})
        self.chosen_key = io_cfg.get("chosen_key", "chosen")
        self.rejected_key = io_cfg.get("rejected_key", "rejected")

        # --------------------------------------------------
        # 3. 构建 pipeline
        # --------------------------------------------------
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

        # freeze VAE
        if getattr(self.pipe, "vae", None) is not None:
            vae = self.pipe.vae
            self.pipe._modules.pop("vae", None)
            object.__setattr__(self.pipe, "vae", vae)
            vae.requires_grad_(False)

        # --------------------------------------------------
        # 4. 切换到训练态（⚠️现在参数已经齐了）
        # --------------------------------------------------
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
        # SFT path: unchanged behavior
        if not self.dpo:
            if inputs is None:
                inputs = self.forward_preprocess(data)
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            loss = self.pipe.training_loss(**models, **inputs)
            return loss
        # DPO path: shared per-sample timesteps + per-sample losses
        chosen_inputs = self.forward_preprocess({
            "prompt": data["prompt"],
            self.chosen_key: data[self.chosen_key]
        }, video_key=self.chosen_key)
        rejected_inputs = self.forward_preprocess({
            "prompt": data["prompt"],
            self.rejected_key: data[self.rejected_key]
        }, video_key=self.rejected_key)

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        # ref_models = {name: getattr(self.ref_pipe, name) for name in self.ref_pipe.in_iteration_models}

        # Determine batch size from inputs
        b = chosen_inputs["input_latents"].shape[0]
        # Sample one timestep per sample, shared across 4 passes
        idx = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (b,), device=self.pipe.scheduler.timesteps.device)
        timesteps = self.pipe.scheduler.timesteps[idx].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        # Current model per-sample losses
        loss_chosen = self.pipe.training_loss(**models, **chosen_inputs, timesteps=timesteps, reduction='none')
        loss_rejected = self.pipe.training_loss(**models, **rejected_inputs, timesteps=timesteps, reduction='none')
        # Reference model per-sample losses (no grad)
        # with torch.no_grad():
        #     ref_chosen = self.ref_pipe.training_loss(**ref_models, **chosen_inputs, timesteps=timesteps, reduction='none')
        #     ref_rejected = self.ref_pipe.training_loss(**ref_models, **rejected_inputs, timesteps=timesteps, reduction='none')

        advantage = (loss_rejected - loss_chosen) # - (ref_rejected - ref_chosen)
        advantage = advantage.clamp(-20, 20)
        dpo_loss = -F.logsigmoid(self.dpo_beta * advantage).mean()

        if torch.distributed.get_rank() == 0:
            for n,t in [("loss_c", loss_chosen), ("loss_r", loss_rejected)]:
                print(n, "shape=", tuple(t.shape), "mean=", float(t.mean()), "std=", float(t.std()),
                    "requires_grad=", t.requires_grad)

            print("mean(delta_policy)=", float((loss_rejected - loss_chosen).mean()))
            print("mean(adv)         =", float(advantage.mean()), "std(adv)=", float(advantage.std()))

            print("lat_diff=", float((chosen_inputs["input_latents"] - rejected_inputs["input_latents"]).abs().mean()))

            with open("adv_chosen.txt", "a") as f:
                f.write(f"{float((loss_chosen).mean())}\n")
            with open("adv_reject.txt", "a") as f:
                f.write(f"{float((loss_rejected).mean())}\n")
            with open("adv_mean.txt", "a") as f:
                f.write(f"{float(advantage.mean())}\n")

        return dpo_loss

