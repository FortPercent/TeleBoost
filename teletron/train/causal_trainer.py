from teletron.models.causwan import CausalDiffusion
from teletron.train import Trainer
from teletron.utils.wan_dataset import TensorDataset, cycle
from teletron.utils.misc import set_seed
import torch.distributed as dist
import dataclasses
from megatron.core import mpu, tensor_parallel
from megatron.core.optimizer import OptimizerConfig
from omegaconf import OmegaConf

import deepspeed
import torch
import time
import os
from tqdm import tqdm
from teletron.utils import (
    print_rank_0,
    print_datetime,
    get_model_config,
    print_rank_last,
    is_last_rank,
    num_floating_point_operations,
    validate_args,
    set_args,
    get_args,
    update_num_microbatches,
    get_num_microbatches,
)
from teletron.train.utils import (
    _initialize_distributed,
    _compile_dependencies,
    set_jit_fusion_options,
    core_transformer_config_from_args,
    forward_step,
    deepspeed_forward_backward,
    _set_random_seed,
    _initialize_tp_communicators,
    calc_params_l2_norm,
    get_grad_norm
)
from teletron.core.parallel_state import get_transformer_model_group
from teletron.train.dataloader import DataloaderMixin
from teletron.models.build import build_model
from teletron.train.checkpoint import CheckPointMixin, unwrap_model, ensure_directory_exists
from teletron.train.lr_scheduler import SchedulerMixin
from teletron.train.telelogger import TeleLoggerMixin
from logging import getLogger
from teletron.datasets.build import build_train_valid_test_datasets
from teletron.core.distributed.distributed_encoder import producer_process
from teletron.models.encoder_registry import get_encoder_name
from teletron.train.consumer_dataloader import create_batch_loader



class DiffusionTrainer:
    def __init__(self, args):
        self.config = args
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # launch_distributed_job()
        # global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        # self.is_main_process = global_rank == 0
        self.is_main_process = True
        self.causal = True # config.causal
        self.disable_tensorboard = getattr(args, 'disable_tensorboard', True)

        # use a random seed for the training
        if args.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            # dist.broadcast(random_seed, src=0)
            args.seed = random_seed.item()

        # set_seed(config.seed + global_rank)
        set_seed(args.seed)

        # Initialize TensorBoard writer
        self.writer = None
        # if self.is_main_process and not self.disable_tensorboard:
        #     # Add timestamp to tensorboard log directory
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     base_tensorboard_dir = getattr(config, 'tensorboard_log_dir', os.path.join(config.logdir, 'tensorboard'))
        #     tensorboard_log_dir = os.path.join(base_tensorboard_dir, f"run_{timestamp}")
            
        #     os.makedirs(tensorboard_log_dir, exist_ok=True)
        #     self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
            
        #     # Log config as text
        #     config_str = OmegaConf.to_yaml(config)
        #     self.writer.add_text('config', config_str, 0)

        self.output_path = args.save

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(args, device=self.device)

        # self.model.generator = fsdp_wrap(
        #     self.model.generator,
        #     sharding_strategy=config.sharding_strategy,
        #     mixed_precision=config.mixed_precision,
        #     wrap_strategy=config.generator_fsdp_wrap_strategy
        # )
        self.model.generator = self.model.generator.to(torch.bfloat16).to(self.device)
        # self.model.text_encoder = fsdp_wrap(
        #     self.model.text_encoder,
        #     sharding_strategy=config.sharding_strategy,
        #     mixed_precision=config.mixed_precision,
        #     wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        # )
        self.model.text_encoder = self.model.text_encoder.to(torch.bfloat16).to(self.device)
        # if not config.no_visualize or config.load_raw_video:
        #     self.model.vae = self.model.vae.to(
        #         device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )

        # world_size = dist.get_world_size() 
        # local_rank = dist.get_rank()
        # if local_rank == 0:
        #     print(f"num of all gpus: {world_size}")

        # Step 3: Initialize the dataloader
        dataset = TensorDataset(args.base_paths, args.metadata_paths)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        # sampler = RandomSampler(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=8)

        # if dist.get_rank() == 0:
        print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p


        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        self.max_grad_norm = 0.5
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        # generator_state_dict = fsdp_state_dict(
        #     self.model.generator)
        generator_state_dict = self.model.generator.state_dict()

        state_dict = {
            "generator": generator_state_dict,
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self, batch):

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        accumulation_steps = getattr(self, "accumulation_steps", 1)

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompt_emb"]
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["latents"].to(self.device, self.dtype)
        else:  # encode raw video to latent
            frames = batch["frames"].to(self.device, self.dtype)
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(frames).to(self.device, self.dtype)

        clean_latent = clean_latent.permute(0, 2, 1, 3, 4)
        image_latent = clean_latent[:, 0:1, ]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent
        )

        # ========== loss / accumulation_steps ==========
        generator_loss = generator_loss / accumulation_steps
        generator_loss.backward()

        if (self.step + 1) % accumulation_steps == 0:
            # generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
            generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.generator.parameters(), self.max_grad_norm)
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad()
        else:
            generator_grad_norm = torch.tensor(0.0, device=self.device)  # dummy value if not stepped

        self.step += 1

        # Step 4: Logging
        if self.is_main_process and self.writer is not None:
            self.writer.add_scalar('loss/generator_loss', generator_loss.item() * accumulation_steps, self.step)
            self.writer.add_scalar('gradient/generator_grad_norm', generator_grad_norm.item(), self.step)

            if log_dict and False:
                for key, value in log_dict.items():
                    if isinstance(value, (int, float, torch.Tensor)):
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        self.writer.add_scalar(f'metrics/{key}', value, self.step)

    def train(self):
        # Set total number of iterations
        total_iterations = 100000
        
        # Create tqdm progress bar
        if self.is_main_process:
            pbar = tqdm(total=total_iterations, desc="Training", 
                    unit="iter", ncols=100, 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        iteration = 0
        while iteration < total_iterations:
            batch = next(self.dataloader)
            self.train_one_step(batch)
            
            if self.config.save is not None and self.step % self.config.save_interval == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if self.writer is not None:
                        self.writer.add_scalar("timing/per_iteration_time", current_time - self.previous_time, self.step)
                    self.previous_time = current_time
                
                # Update progress bar
                pbar.update(1)
            
            iteration += 1
        
        # Close progress bar
        if self.is_main_process:
            pbar.close()

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        sampled_noise = torch.randn(
            [batch_size, 21, 16, 60, 104], device="cuda", dtype=self.dtype
        )
        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def __del__(self):
        # Close tensorboard writer when trainer is destroyed
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()




class CausalTrainer(Trainer):
    def __init__(self, args, dataset_provider_func=None):
        super().__init__(args)
    
    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        add_encoder=True,
        add_decoder=True,
        parallel_output=True,
    ):
        args = get_args()
        # cfg = core_transformer_config_from_args(args)
        return build_model(args.model, args)

    # def setup_model_and_optimizer(self, model_type, 
    #                               no_wd_decay_cond=None, 
    #                               scale_lr_cond=None, 
    #                               lr_mult=1.0):
    #     args = get_args()
    #     # timers = get_timers()
    #     assert args.global_batch_size == args.micro_batch_size * mpu.get_data_parallel_world_size()
    #     # timers = get_timers()
    #     if args.use_zero2:
    #         model = self.get_model(model_type, wrap_with_ddp=False)
    #     else:
    #         model = self.get_model(model_type)
    #     # self.model.generator = self.model.generator.to(torch.bfloat16)
    #     unwrapped_model = unwrap_model(model)
    #     kwargs = {}
    #     for f in dataclasses.fields(OptimizerConfig):
    #         if hasattr(args, f.name):
    #             kwargs[f.name] = getattr(args, f.name)
    #     config = OptimizerConfig(**kwargs)
    #     config.timers = None
    #     if args.use_zero2:
    #         deepspeed.init_distributed()
    #         optimizer = self.get_optimizer_for_zero2(config, model, no_wd_decay_cond,
    #                                     scale_lr_cond, lr_mult)
    #     else:
    #         optimizer = self.get_optimizer(config, model, no_wd_decay_cond,
    #                                     scale_lr_cond, lr_mult)

    #     opt_param_scheduler = self.get_optimizer_param_scheduler(optimizer)

    #     if args.load is not None or args.pretrained_checkpoint is not None:
    #         # timers('load-checkpoint', log_level=0).start(barrier=True)
    #         args.iteration, args.num_floating_point_operations_so_far = self.load_checkpoint(
    #             model, optimizer, opt_param_scheduler, strict=True)
    #         # timers('load-checkpoint').stop(barrier=True)
    #         # timers.log(['load-checkpoint'])
    #     else:
    #         args.iteration = 0
    #         args.num_floating_point_operations_so_far = 0
    #         args.last_microbatch_size_index = None

    #     # get model without FP16 and/or DDP wrappers
    #     if args.iteration == 0 and len(unwrapped_model) == 1 \
    #         and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
    #         print_rank_0("Initializing ICT from pretrained BERT model")
    #         unwrapped_model[0].init_state_dict_from_bert()
    #         if args.fp16:
    #             optimizer.reload_model_params()

    #     return model, optimizer, opt_param_scheduler
