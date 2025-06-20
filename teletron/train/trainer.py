import torch
import torch.distributed as dist
import dataclasses
import time
import random
import numpy as np
import sys
import gc
from typing import Callable, Dict, List, Optional
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer.module import MegatronModule, Float16Module
from megatron.core.enums import ModelType
from megatron.core.distributed import finalize_model_grads
from megatron.core import mpu, tensor_parallel, dist_checkpointing
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import ( OptimizerConfig, 
                                     _get_param_groups,
                                     _update_min_and_max_lr_in_param_groups,
                                     _get_megatron_optimizer_based_on_param_groups,
                                     ChainedOptimizer,)
        
from vast.train.samplers import build_sampler as build_sampler_vast
from vast.datasets import DefaultCollator
from vast.datasets.datasets.build import build_dataset as build_dataset_vast
from teletron.models.wan.wan_producer import producer_process
from teletron.utils.scheduler import OptimizerParamScheduler
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
from teletron.datasets.build import build_dataset
from teletron.train.utils import (_initialize_distributed,
                                  _compile_dependencies,
                                  set_jit_fusion_options,
                                  core_transformer_config_from_args,
                                  load_config_vast,
                                  get_train_valid_test_num_samples,
                                  forward_step,
                                  _set_random_seed,
                                  _initialize_tp_communicators,
                                  update_train_iters,
                                  )
from teletron.utils.checkpoint import ( _load_base_checkpoint,
                                       read_metadata,
                                       get_checkpoint_name,
                                       get_rng_state,
                                       get_checkpoint_tracker_filename,
                                       ensure_directory_exists,
                                       checkpoint_exists,
                                       get_distributed_optimizer_checkpoint_name,
                                        )
from teletron.core.parallel_state import get_transformer_model_group
from teletron.core.data_loader import build_pretraining_data_loader
from teletron.datasets.vast_dataset.hunyuan_dataset_config import HunyuanVideoDatasetConfig
from teletron.datasets.vast_dataset.hunyuanvideo_dataset_builder import HunyuanVideoDatasetBuilder
from teletron.models.build import build_model
from logging import getLogger
logger = getLogger(__name__)
_TRAIN_START_TIME = time.time()
ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model

def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

class Trainer:
    def __init__(self, 
                 args,
                 dataset_provide_func=None, 
                 ):
        # args = args_tmp

        self.initialize_megatron(args)
        set_jit_fusion_options()
        transformer_group = get_transformer_model_group()
        if transformer_group is None:
            train_ds, _, _ = self.build_train_valid_test_datasets()
            producer_process(
                rank=dist.get_rank(), 
                world_size=dist.get_world_size(),
                build_train_valid_test_data_iterators=self.build_train_valid_test_data_iterators, 
                train_ds=train_ds,
            )
            exit()        
        global _TRAIN_START_TIME
        start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                        dtype=torch.double,
                                        device='cuda')
        torch.distributed.all_reduce(start_time_tensor,
                                    op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
            time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')
        

        self.model, self.optimizer, self.scheduler = \
                                self.setup_model_and_optimizer(args.model_type)
                
        self.train_itrt, self.valid_itrt, self.test_itrt = \
                                self.get_iterator(len(self.model), dataset_provide_func)
        
        self.config = get_model_config(self.model[0])


    def setup_model_and_optimizer(self,  
                                  model_type,
                                  no_wd_decay_cond=None,
                                  scale_lr_cond=None,
                                  lr_mult=1.0):

        
        args = get_args()
        # timers = get_timers()
        # set_jit_fusion_options()
        
        assert args.global_batch_size == args.micro_batch_size * mpu.get_data_parallel_world_size()
        # timers = get_timers()
        model = self.get_model(model_type)
        unwrapped_model = unwrap_model(model)
        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)
        config.timers = None
        optimizer = self.get_optimizer(config, model, no_wd_decay_cond,
                                        scale_lr_cond, lr_mult)
        
        opt_param_scheduler = self.get_optimizer_param_scheduler(optimizer)
        if args.load is not None or args.pretrained_checkpoint is not None:
            # timers('load-checkpoint', log_level=0).start(barrier=True)
            args.iteration, args.num_floating_point_operations_so_far = self.load_checkpoint(
                model, optimizer, opt_param_scheduler, strict=True)
            # timers('load-checkpoint').stop(barrier=True)
            # timers.log(['load-checkpoint'])
        else:
            args.iteration = 0
            args.num_floating_point_operations_so_far = 0

        # get model without FP16 and/or DDP wrappers
        if args.iteration == 0 and len(unwrapped_model) == 1 \
            and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
            print_rank_0("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()
            if args.fp16:
                optimizer.reload_model_params()

        return model, optimizer, opt_param_scheduler

    def get_optimizer_param_scheduler(self, optimizer):
        """Build the learning rate scheduler."""
        args = get_args()

        # Iteration-based training.
        if args.train_iters:
            if args.lr_decay_iters is None:
                args.lr_decay_iters = args.train_iters
            lr_decay_steps = args.lr_decay_iters * args.global_batch_size
            wd_incr_steps = args.train_iters * args.global_batch_size
            if args.lr_warmup_fraction is not None:
                lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
            else:
                lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
        # Sample-based training.
        elif args.train_samples:
            # We need to set training iters for later use. Technically
            # we need to adjust the training samples too (due to last
            # batch being incomplete) but we leave it as is for now.
            update_train_iters(args)
            if args.lr_decay_samples is None:
                args.lr_decay_samples = args.train_samples
            lr_decay_steps = args.lr_decay_samples
            wd_incr_steps = args.train_samples
            if args.lr_warmup_fraction is not None:
                lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
            else:
                lr_warmup_steps = args.lr_warmup_samples
        else:
            raise Exception(
                'either train-iters or train-samples should be provided.')

        opt_param_scheduler = OptimizerParamScheduler(
            optimizer,
            init_lr=args.lr_warmup_init,
            max_lr=args.lr,
            min_lr=args.min_lr,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_decay_style=args.lr_decay_style,
            start_wd=args.start_weight_decay,
            end_wd=args.end_weight_decay,
            wd_incr_steps=wd_incr_steps,
            wd_incr_style=args.weight_decay_incr_style,
            use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
            override_opt_param_scheduler=args.override_opt_param_scheduler)

        return opt_param_scheduler



    def get_scheduler(self, optimizer):
        args = get_args()
        # Iteration-based training.
        if args.train_iters:
            if args.lr_decay_iters is None:
                args.lr_decay_iters = args.train_iters
            lr_decay_steps = args.lr_decay_iters * args.global_batch_size
            wd_incr_steps = args.train_iters * args.global_batch_size
            if args.lr_warmup_fraction is not None:
                lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
            else:
                lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
        # Sample-based training.
        elif args.train_samples:
            # We need to set training iters for later use. Technically
            # we need to adjust the training samples too (due to last
            # batch being incomplete) but we leave it as is for now.
            update_train_iters(args)
            if args.lr_decay_samples is None:
                args.lr_decay_samples = args.train_samples
            lr_decay_steps = args.lr_decay_samples
            wd_incr_steps = args.train_samples
            if args.lr_warmup_fraction is not None:
                lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
            else:
                lr_warmup_steps = args.lr_warmup_samples
        else:
            raise Exception(
                'either train-iters or train-samples should be provided.')

        opt_param_scheduler = OptimizerParamScheduler(
            optimizer,
            init_lr=args.lr_warmup_init,
            max_lr=args.lr,
            min_lr=args.min_lr,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_decay_style=args.lr_decay_style,
            start_wd=args.start_weight_decay,
            end_wd=args.end_weight_decay,
            wd_incr_steps=wd_incr_steps,
            wd_incr_style=args.weight_decay_incr_style,
            use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
            override_opt_param_scheduler=args.override_opt_param_scheduler)

        return opt_param_scheduler


    def get_optimizer(self, config: OptimizerConfig, 
                      model: List[MegatronModule], 
                      no_weight_decay_cond: Optional[Callable] = None, 
                      scale_lr_cond: Optional[Callable] = None, 
                      lr_mult: float = 1.0):
        
        """Retrieve the Megatron optimizer for model chunks.

        We use separate optimizers for expert parameters and non-expert parameters.

        Args:
            config (OptimizerConfig): optimizer configuration object.
            model_chunks (List[MegatronModule]): model chunks to get optimizer for.
            no_weight_decay_cond (func, optional): function to determine whether a parameter
                should not perform weight decay. Defaults to None.
            scale_lr_cond (func, optional): function to determine whether a parameter
                should have a scaled learning rate. Defaults to None.
            lr_mult (float, optional): learning rate multiplier for parameters that
                satisfy scale_lr_cond. Defaults to 1.0.

        Returns:
            Instance of MegatronOptimizer.
        """

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(f'Setting up optimizer with {config}')

        # Collect param groups.
        param_groups = _get_param_groups(
            model,
            no_weight_decay_cond,
            scale_lr_cond,
            lr_mult,
            use_decoupled_learning_rate=config.decoupled_lr is not None,
        )
        param_groups = _update_min_and_max_lr_in_param_groups(
            param_groups,
            lr=config.lr,
            min_lr=config.min_lr,
            decoupled_lr=config.decoupled_lr,
            decoupled_min_lr=config.decoupled_min_lr,
        )

        # Collect grad buffers for distributed optimizer.
        per_model_buffers = {}
        per_model_ep_buffers = {}
        for model_idx, model_chunk in enumerate(model):
            if hasattr(model_chunk, 'buffers'):
                per_model_buffers[model_idx] = model_chunk.buffers
                per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers

        # Split param groups into dense and MoE params (since data-parallel groups for MoE
        # parameters can be different with expert parallelism).
        dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))
        moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))

        # Create optimizers.
        model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())
        optimizers = [
            _get_megatron_optimizer_based_on_param_groups(
                config,
                param_groups=dense_param_groups,
                per_model_buffers=per_model_buffers,
                data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
                data_parallel_group_idx=model_parallel_rank,
            )
        ]
        if len(moe_param_groups) > 0:
            model_parallel_world_size = torch.distributed.get_world_size(mpu.get_model_parallel_group())
            expert_parallel_rank = mpu.get_expert_model_parallel_rank()
            optimizers.append(
                _get_megatron_optimizer_based_on_param_groups(
                    config,
                    param_groups=moe_param_groups,
                    per_model_buffers=per_model_ep_buffers,
                    data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                    data_parallel_group_gloo=mpu.get_data_modulo_expert_parallel_group_gloo(),
                    data_parallel_group_idx=expert_parallel_rank * model_parallel_world_size
                    + model_parallel_rank,
                )
            )

        if len(optimizers) == 1:
            return optimizers[0]

        return ChainedOptimizer(optimizers)


    def model_provider(self,
                        pre_process=True,
                        post_process=True,
                        add_encoder=True,
                        add_decoder=True,
                        parallel_output=True):
        args=get_args()
        cfg = core_transformer_config_from_args(args)
        # breakpoint()
        return build_model(args.model, cfg)


    def get_model(self,
                   model_type=ModelType.encoder_or_decoder, 
                   wrap_with_ddp=True):
        args = get_args()
        args.model_type = model_type
        if mpu.get_pipeline_model_parallel_world_size() > 1 and \
            args.virtual_pipeline_model_parallel_size is not None:
            assert model_type != ModelType.encoder_and_decoder, \
                "Interleaved schedule not supported for model with both encoder and decoder"
            model = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                this_model = self.model_provider(
                    pre_process=pre_process,
                    post_process=post_process
                )
                this_model.model_type = model_type
                model.append(this_model)
        else:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            add_encoder = True
            add_decoder = True
            if model_type == ModelType.encoder_and_decoder:
                if mpu.get_pipeline_model_parallel_world_size() > 1:
                    assert args.pipeline_model_parallel_split_rank is not None, \
                        "Split rank needs to be specified for model with both encoder and decoder"
                    rank = mpu.get_pipeline_model_parallel_rank()
                    split_rank = args.pipeline_model_parallel_split_rank
                    world_size = mpu.get_pipeline_model_parallel_world_size()
                    pre_process = rank == 0 or rank == split_rank
                    post_process = (rank == (split_rank - 1)) or (
                            rank == (world_size - 1))
                    add_encoder = mpu.is_pipeline_stage_before_split()
                    add_decoder = mpu.is_pipeline_stage_after_split()
                model = self.model_provider(
                    pre_process=pre_process,
                    post_process=post_process,
                    add_encoder=add_encoder,
                    add_decoder=add_decoder)
            else:
                model = self.model_provider(
                    pre_process=pre_process,
                    post_process=post_process
                )
            # breakpoint()
            model.model_type = model_type

        # breakpoint()

        if not isinstance(model, list):
            model = [model]

        # breakpoint()
        # Set tensor model parallel attributes if not set.
        # Only parameters that are already tensor model parallel have these
        # attributes set for them. We should make sure the default attributes
        # are set for all params so the optimizer can use them.
        for model_module in model:
            for param in model_module.parameters():
                tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

        # GPU allocation.
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

        # Fp16 conversion.
        if args.fp16 or args.bf16:
            model = [Float16Module(module=model_module, config=model_module.config) for model_module in model]
        #import ipdb; ipdb.set_trace()
        # breakpoint()
        if wrap_with_ddp:
            config = get_model_config(model[0])
            # breakpoint()
            model = [DDP(config,
                        model_chunk,
                        data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                        expert_data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                        accumulate_allreduce_grads_in_fp32=args.accumulate_allreduce_grads_in_fp32,
                        overlap_grad_reduce=args.overlap_grad_reduce,
                        use_distributed_optimizer=args.use_distributed_optimizer,
                        # Turn off bucketing for model_chunk 2 onwards, since communication for these
                        # model chunks is overlapped with compute anyway.
                        disable_bucketing=(model_chunk_idx > 0),
                        check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad)
                    for (model_chunk_idx, model_chunk) in enumerate(model)]

            # Broadcast params from data parallel src rank to other data parallel ranks.
            if args.data_parallel_random_init:
                for model_module in model:
                    model_module.broadcast_params()

        return model
        

    def load_checkpoint(self, model, optimizer, opt_param_scheduler, load_arg='load', strict=True):
        """Load a model checkpoint and return the iteration.
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` of the checkpoint match the names of
            parameters and buffers in model.
        """
        args = get_args()
        load_dir = getattr(args, load_arg)

        # Finetuning directories
        pretrained_dir = getattr(args,'pretrained_checkpoint', None)
        if pretrained_dir is not None and not checkpoint_exists(load_dir):
            print_rank_0(f'Checkpoint file not found in load directory {load_dir} attempting to finetune with checkpoint in {pretrained_dir}')
            load_dir = pretrained_dir
            if not checkpoint_exists(load_dir):
                raise FileNotFoundError("No checkpoint found in load directory or pretrained directory")
            args.finetune = True


        model = unwrap_model(model)

        load_kwargs = {}
        is_dist_ckpt = False
        if args.auto_detect_ckpt_format or args.use_dist_ckpt:
            state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=True, exit_on_missing_checkpoint=args.exit_on_missing_checkpoint)
            is_dist_ckpt = dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
            if is_dist_ckpt:
                ckpt_tp_pp = (state_dict['args'].tensor_model_parallel_size, state_dict['args'].pipeline_model_parallel_size)
                run_tp_pp = (mpu.get_tensor_model_parallel_world_size(), mpu.get_pipeline_model_parallel_world_size())
                mismatch_msg = "(TP, PP) mismatch after resume ({} vs {} from checkpoint)".format(ckpt_tp_pp, run_tp_pp)

                if ckpt_tp_pp == run_tp_pp and not getattr(state_dict['args'], 'no_save_rng', False):
                    rng_state = get_rng_state(True)  # we can load the rng state
                else:
                    rng_state = None
                    print_rank_0("{}: RNG state will be ignored".format(mismatch_msg))

                # TODO: add DistributedOptimizer support for differing TPxPP
                if ckpt_tp_pp != run_tp_pp and not release and not args.finetune and not args.no_load_optim and args.use_distributed_optimizer:
                    raise RuntimeError("{}: not supported for DistributedOptimizer".format(mismatch_msg))

                optim_sd_kwargs = dict(is_loading=True)
                if args.use_distributed_optimizer:
                    optim_sd_kwargs['sharding_type'] = ('fully_sharded_bucket_space'
                                                        if getattr(state_dict['args'], 'ckpt_fully_parallel_save', False)
                                                        else 'dp_zero_gather_scatter')
                load_kwargs['sharded_state_dict'] = self.generate_state_dict(args, model, optimizer, opt_param_scheduler,
                                                                        rng_state, args.use_dist_ckpt, optim_sd_kwargs=optim_sd_kwargs)
                load_kwargs['exit_on_missing_checkpoint'] = args.exit_on_missing_checkpoint

        state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=False, **load_kwargs)

        # Checkpoint not loaded.
        if state_dict is None:
            # Iteration and num_floating_point_operations_so_far default to 0.
            return 0, 0

        # Set checkpoint version.
        # set_checkpoint_version(state_dict.get('checkpoint_version', 0))

        # Set iteration.
        if args.finetune or release:
            iteration = 0
        else:
            try:
                iteration = state_dict['iteration']
            except KeyError:
                try:  # Backward compatible with older checkpoints
                    iteration = state_dict['total_iters']
                except KeyError:
                    print_rank_0('A metadata file exists but unable to load '
                                'iteration from checkpoint {}, exiting'.format(checkpoint_name))
                    sys.exit()
        num_floating_point_operations_so_far = state_dict.get('num_floating_point_operations_so_far', 0)

        # Check arguments.
        assert args.consumed_train_samples == 0
        assert args.consumed_valid_samples == 0
        if 'args' in state_dict and not args.finetune:
            checkpoint_args = state_dict['args']
            # check_checkpoint_args(checkpoint_args)
            args.consumed_train_samples = getattr(checkpoint_args,
                                                'consumed_train_samples', 0)
            update_num_microbatches(consumed_samples=args.consumed_train_samples)
            args.consumed_valid_samples = getattr(checkpoint_args,
                                                'consumed_valid_samples', 0)
        else:
            print_rank_0('could not find arguments in the checkpoint ...')

        # Model.
        strict = False if args.retro_add_retriever else strict

        if args.lora == True:
            raise NotImplementedError('Lora not implement yet')
            from peft import get_peft_model, LoraConfig, TaskType
            from teletron.models.wan.light_pipeline import find_lora_target_modules
            base_model_path = args.lora_base_model_path
            target_modules = find_lora_target_modules(model[0], args.lora_target_modules)
            lora_config = LoraConfig(
                r=args.lora_rank,                             # LoRA rank
                lora_alpha=args.lora_alpha,                   # Scaling factor
                # target_modules=["q", "v","o","k"],  # Layer names to apply LoRA to
                target_modules=target_modules, # Layer names to apply LoRA to
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                task_type= TaskType[args.lora_task_type]    # or SEQ_CLS, TOKEN_CLS etc.
            )
            if len(model) == 1:
                model[0].load_state_dict(torch.load(base_model_path))
                model[0]=get_peft_model(model[0], lora_config)
                model[0].load_state_dict(state_dict['model'], strict=strict)
            else:
                for i in range(len(model)):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    model[i].load_state_dict(torch.load(base_model_path), strict=strict)
                    model[i]=get_peft_model(model[i], lora_config)
                    model[i].load_state_dict(state_dict['model%d' % i], strict=strict)
        else:
            if len(model) == 1:
                model[0].load_state_dict(state_dict['model'], strict=strict)
            else:
                for i in range(len(model)):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

        # Fix up query/key/value matrix ordering if needed.
        # checkpoint_version = get_checkpoint_version()
        # print_rank_0(f' checkpoint version {checkpoint_version}')
        # fix_query_key_value_ordering(model, checkpoint_version)

        # Optimizer.
        if not release and not args.finetune and not args.no_load_optim:
            try:
                # Load state dict.
                if optimizer is not None:
                    optimizer.load_state_dict(state_dict['optimizer'])

                # Load distributed optimizer's custom parameter state.
                # For distributed checkpoint it's already loaded in load_state_dict above
                if args.use_distributed_optimizer and not is_dist_ckpt:
                    tracker_filename = get_checkpoint_tracker_filename(load_dir)
                    iteration, release = read_metadata(tracker_filename)
                    model_checkpoint_name = \
                        get_checkpoint_name(load_dir, iteration, release)
                    optim_checkpoint_name = \
                        get_distributed_optimizer_checkpoint_name(
                            model_checkpoint_name)
                    optimizer.load_parameter_state(optim_checkpoint_name)

                # Load scheduler.
                if opt_param_scheduler is not None:
                    if 'lr_scheduler' in state_dict: # backward compatbility
                        opt_param_scheduler.load_state_dict(state_dict['lr_scheduler'])
                    else:
                        opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
            except KeyError:
                print_rank_0('Unable to load optimizer from checkpoint {}. '
                            'Specify --no-load-optim or --finetune to prevent '
                            'attempting to load the optimizer state, '
                            'exiting ...'.format(checkpoint_name))
                sys.exit()
        else:
            if (args.fp16 or args.bf16) and optimizer is not None:
                optimizer.reload_model_params()

        # rng states.
        if not release and not args.finetune and not args.no_load_rng:
            try:
                if 'rng_state' in state_dict:
                    # access rng_state for data parallel rank
                    if args.data_parallel_random_init:
                        rng_state = state_dict['rng_state'][mpu.get_data_parallel_rank()]
                    else:
                        rng_state = state_dict['rng_state'][0]
                    random.setstate(rng_state['random_rng_state'])
                    np.random.set_state(rng_state['np_rng_state'])
                    torch.set_rng_state(rng_state['torch_rng_state'])
                    torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                    # Check for empty states array
                    if not rng_state['rng_tracker_states']:
                        raise KeyError
                    tensor_parallel.get_cuda_rng_tracker().set_states(
                        rng_state['rng_tracker_states'])
                else:  # backward compatability
                    random.setstate(state_dict['random_rng_state'])
                    np.random.set_state(state_dict['np_rng_state'])
                    torch.set_rng_state(state_dict['torch_rng_state'])
                    torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                    # Check for empty states array
                    if not state_dict['rng_tracker_states']:
                        raise KeyError
                    tensor_parallel.get_cuda_rng_tracker().set_states(
                        state_dict['rng_tracker_states'])
            except KeyError:
                print_rank_0('Unable to load rng state from checkpoint {}. '
                            'Specify --no-load-rng or --finetune to prevent '
                            'attempting to load the rng state, '
                            'exiting ...'.format(checkpoint_name))
                sys.exit()
        # from .global_vars import _GLOBAL_ARGS
        args.last_micro_batch_access_index = state_dict["last_microbatch_size_index"]

        # Some utilities want to load a checkpoint without distributed being initialized
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        print_rank_0(f'  successfully loaded checkpoint from {load_dir} '
                    f'[ t {mpu.get_tensor_model_parallel_rank()}, '
                    f'p {mpu.get_pipeline_model_parallel_rank()} ] '
                    f'at iteration {iteration}')

        return iteration, num_floating_point_operations_so_far


    def save_checkpoint(self, iteration, model, optimizer, opt_param_scheduler,
                    num_floating_point_operations_so_far):
        """Save a model checkpoint."""
        args = get_args()

        # Only rank zero of the data parallel writes to the disk.
        model = unwrap_model(model)

        ckpt_format = args.dist_ckpt_format if args.use_dist_ckpt else 'torch'
        print_rank_0('saving checkpoint at iteration {:7d} to {} in {} format'.format(
            iteration, args.save, ckpt_format))

        # Collect rng state across data parallel ranks.
        rng_state = get_rng_state(args.use_dist_ckpt)

        # Checkpoint name.
        checkpoint_name = get_checkpoint_name(args.save, iteration, return_base_dir=args.use_dist_ckpt)

        # Save distributed optimizer's custom parameter state.
        if args.use_distributed_optimizer and not args.no_save_optim and optimizer is not None and not args.use_dist_ckpt:
            optim_checkpoint_name = \
                get_distributed_optimizer_checkpoint_name(checkpoint_name)
            ensure_directory_exists(optim_checkpoint_name)
            optimizer.save_parameter_state(optim_checkpoint_name)

        # Collect args, model, RNG.
        if not torch.distributed.is_initialized() \
                or mpu.get_data_modulo_expert_parallel_rank() == 0 \
                or args.use_dist_ckpt:

            optim_sd_kwargs = {}
            if args.use_dist_ckpt and args.use_distributed_optimizer:
                optim_sd_kwargs['sharding_type'] = ('fully_sharded_bucket_space'
                                                    if args.ckpt_fully_parallel_save
                                                    else 'dp_zero_gather_scatter')
                print_rank_0(f'Storing distributed optimizer sharded state of type {optim_sd_kwargs["sharding_type"]}')
            state_dict = self.generate_state_dict(args, model, optimizer, opt_param_scheduler, rng_state,
                                            args.use_dist_ckpt, iteration, optim_sd_kwargs=optim_sd_kwargs)

            state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far
            if args.use_dist_ckpt:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    ensure_directory_exists(checkpoint_name,
                                            check_parent=False)
                dist_checkpointing.save(state_dict, checkpoint_name, (args.dist_ckpt_format, 1))

            else:
                # Save.
                ensure_directory_exists(checkpoint_name)
                torch.save(state_dict, checkpoint_name)

        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        print_rank_0('  successfully saved checkpoint at iteration {:7d} to {}' \
                    .format(iteration, args.save))

        # And update the latest iteration
        if not torch.distributed.is_initialized() \
        or torch.distributed.get_rank() == 0:
            tracker_filename = get_checkpoint_tracker_filename(args.save)
            with open(tracker_filename, 'w') as f:
                f.write(str(iteration))

        # Wait so everyone is done (not necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


    def generate_state_dict(self, args, model, optimizer, opt_param_scheduler,
                        rng_state, use_dist_ckpt=False, iteration=None,sampler=None,
                        optim_sd_kwargs=None):
        # Arguments, iteration, and model.
        state_dict = {}
        state_dict['args'] = args
        state_dict['checkpoint_version'] = 3.0
        if iteration is not None:
            state_dict['iteration'] = iteration
        # save bucketSample last_microbatch_size_index
        state_dict["last_microbatch_size_index"]=args.last_microbatch_size_index

        if len(model) == 1:
            state_dict['model'] = (model[0].sharded_state_dict()
                                if use_dist_ckpt else
                                model[0].state_dict_for_save_checkpoint())
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict['model%d' % i] = (
                    model[i].sharded_state_dict()
                    if use_dist_ckpt else
                    model[i].state_dict_for_save_checkpoint())
        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                state_dict['optimizer'] = (optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
                                        if use_dist_ckpt else
                                        optimizer.state_dict())
            if opt_param_scheduler is not None:
                state_dict['opt_param_scheduler'] = \
                    opt_param_scheduler.state_dict()
        # RNG states.
        if not args.no_save_rng:
            state_dict["rng_state"] = rng_state
        return state_dict


    def get_iterator(self, 
                     len_model:int, 
                     train_valid_test_dataset_provider,
                     ):
        args = get_args()
        if args.virtual_pipeline_model_parallel_size is not None:
            train_itrt = []
            valid_itrt = []
            test_itrt = []
            for i in range(len_model):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                iterators = self.build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider)
                train_itrt.append(iterators[0])
                valid_itrt.append(iterators[1])
                test_itrt.append(iterators[2])
        else:
            train_ds, valid_ds, test_ds = self.build_train_valid_test_datasets()
            train_itrt, valid_itrt, test_itrt \
                = self.build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider, train_ds_prev=train_ds)
        return train_itrt, valid_itrt, test_itrt
            

    def build_train_valid_test_datasets(self, dp_rank=None, dp_size=None):
        """Build pretraining datasets."""
        train_valid_test_num_samples = get_train_valid_test_num_samples()
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2]))
        args = get_args()

        print_rank_0("> building train, validation, and test datasets for multimodal ...")

        if args.dataset_type == "FakeDataset" or args.dataset_type == "KoalaDataset":
            train_ds = build_dataset(args.dataset_type)
            valid_ds = None
            test_ds = None
        elif args.dataset_type == "VastDataset": 
            global_config = load_config_vast()
            train_ds_config = global_config.dataloaders.train
            eval_ds_config = global_config.dataloaders.get("eval", None)
            ds_config = HunyuanVideoDatasetConfig(
                train_ds_config=train_ds_config,
                eval_ds_config=eval_ds_config
            )
            dataset = build_dataset_vast(train_ds_config.dataset)
            train_ds, valid_ds, test_ds = HunyuanVideoDatasetBuilder(
                dataset,
                train_valid_test_num_samples,
                lambda: True,
                ds_config,
            ).build()

        elif args.dataset_type == "BucketDataset": 
            global_config = load_config_vast()
            assert global_config.sampler.type == "BucketVariableBatchSampler"
            assert args.dataloader_type == 'external', "BucketDataset use cumstomed dataloader"
            assert args.task_type == "wan_i2v_bucket", "BucketDataset is only supported for t2i_wanvae task"
            print_rank_0("Warning: The `args.micro_batch_size` and `seed` from vast dataset config will NOT BE USED when use BucketDataset.")

            dataset = build_dataset_vast(global_config.dataset)
            if dp_rank is None or dp_size is None:
                sampler = build_sampler_vast(
                    global_config.sampler,
                    dataset=dataset, 
                    rank=mpu.get_data_parallel_rank(),
                    num_replicas=mpu.get_data_parallel_world_size(),
                    seed=args.seed
                )
            else:
                sampler = build_sampler_vast(
                    global_config.sampler,
                    dataset=dataset, 
                    rank=dp_rank,
                    num_replicas=dp_size,
                    seed=args.seed
                )
            # 使用 iteration 和 num_replicas / world size 计算，近似最后访问的索引
            # if iteration is None:
            sampler.last_micro_batch_access_index = args.iteration * sampler.num_replicas
            # else:
                # sampler.last_micro_batch_access_index = iteration * sampler.num_replicas

            collator = DefaultCollator(is_equal=True)
            train_ds = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=collator,
                num_workers=global_config.dataloader.num_workers
            )
            train_ds = iter(train_ds)
            valid_ds = None
            test_ds = None
        else:
            raise NotImplementedError

        print_rank_0("> finished creating multimodal datasets ...")

        return train_ds, valid_ds, test_ds

    def build_train_valid_test_data_loaders(self,
                                            is_tp_first = None,
                                            dp_rank = None, 
                                            dp_size = None,  
                                            train_ds_prev = None):
        """Build pretraining data loaders."""

        args = get_args()

        (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

        print_rank_0('> building train, validation, and test datasets ...')

        if args.iteration > 0 and args.consumed_train_samples == 0:
            assert args.train_samples is None, \
                'only backward compatiblity support for iteration-based training'
            args.consumed_train_samples = args.iteration * args.global_batch_size
        if args.iteration > 0 and args.consumed_valid_samples == 0:
            if args.train_samples is None:
                args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                    args.eval_iters * args.global_batch_size

        # Rely on distributed-aware core datasets, temporary
        # is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)
        # print(f"cs rank: {dist.get_rank()}, is_distributed: {is_distributed}")

        # Construct the data pipeline
        # if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:
        # if is_tp_fist is None:
        #     is_tp_fist = (mpu.get_tensor_model_parallel_rank() == 0)
        
        # if is_distributed or is_tp_fist:
            # Build datasets.
        if train_ds_prev is not None:
            train_ds = train_ds_prev
            valid_ds = None
            test_ds = None
        else:
            train_ds, valid_ds, test_ds = self.build_train_valid_test_datasets()
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples, dp_rank, dp_size )
        if args.skip_train:
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0, dp_rank, dp_size )
        else:
            valid_dataloader = build_pretraining_data_loader(
                valid_ds, args.consumed_valid_samples, dp_rank, dp_size )
        test_dataloader = build_pretraining_data_loader(test_ds, 0,  dp_rank, dp_size )

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        flags = torch.tensor(
            [int(do_train), int(do_valid), int(do_test)],
            dtype=torch.long, device='cuda')
        # else:
        #     flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

        if dp_rank is None or dp_size is None:
            torch.distributed.broadcast(flags, 0)

        args.do_train = getattr(args, "do_train", False) or flags[0].item()
        args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
        args.do_test = getattr(args, "do_test", False) or flags[2].item()

        return train_dataloader, valid_dataloader, test_dataloader


    def build_train_valid_test_data_iterators(self,
                    is_tp_first = None,
                    dp_rank = None, 
                    dp_size = None, 
                    train_ds_prev=None):
        """Build pretraining data iterators."""

        args = get_args()

        # Build loaders.
        print("Building loaders.")
        train_dataloader, valid_dataloader, test_dataloader = \
            self.build_train_valid_test_data_loaders(
                is_tp_first, dp_rank, dp_size, train_ds_prev)
        
        # Build iterators.
        print("Building iterators.")
        dl_type = args.dataloader_type

        assert dl_type in ['single', 'cyclic', 'external']

        def _get_iterator(dataloader_type, dataloader):
            """Return dataset iterator."""
            if dataloader_type == "single":
                return iter(dataloader)
            elif dataloader_type == "cyclic":
                return iter(cyclic_iter(dataloader))
            elif dataloader_type == "external":
                # External dataloader is passed through. User is expected to define how to iterate.
                return dataloader
            else:
                raise RuntimeError("unexpected dataloader type")

        if train_dataloader is not None:
            train_data_iterator = _get_iterator(dl_type, train_dataloader)
        else:
            train_data_iterator = None

        if valid_dataloader is not None:
            valid_data_iterator = _get_iterator(dl_type, valid_dataloader)
        else:
            valid_data_iterator = None

        if test_dataloader is not None:
            test_data_iterator = _get_iterator(dl_type, test_dataloader)
        else:
            test_data_iterator = None

        return train_data_iterator, valid_data_iterator, test_data_iterator


    def initialize_megatron(self,
                            args,
        ):

        if args.distributed_vae:
            args.world_size -= args.distributed_vae_world_size
            args.dit_world_size = args.world_size

        validate_args(args)
        set_args(args)

        if args.distributed_vae:
            args.world_size += args.distributed_vae_world_size
        def finish_mpu_init():
            args = get_args()
            _initialize_distributed()
            if args.rank == 0:
                print("> setting random seeds to {} ...".format(args.seed))

            from teletron.core.parallel_state import get_transformer_model_group
            isDiTRank = get_transformer_model_group()
            if isDiTRank is not None:
                _set_random_seed(args.seed, args.data_parallel_random_init)
        args = get_args()
        
        if args.lazy_mpu_init:
            args.use_cpu_initialization = True
            # delayed initialization of DDP-related stuff
            # We only set basic DDP globals
            mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
            # and return function for external DDP manager
            # to call when it has DDP initialized
            mpu.set_tensor_model_parallel_rank(args.rank)
            return finish_mpu_init
        else:
            # Megatron's MPU is the master. Complete initialization right away.
            finish_mpu_init()
            # Autoresume.
            # _init_autoresume()
            # Compile dependencies.
            from teletron.core.parallel_state import get_transformer_model_group
            isConsumerRank = get_transformer_model_group()
            if isConsumerRank is not None:
                _compile_dependencies()
            if args.tp_comm_overlap:
                _initialize_tp_communicators()
            # No continuation function
            return None
    

    def pretrain(self, 
              forward_step_func=forward_step,
              process_non_loss_data_func=None,
              ):
        args = get_args()

        if args.distributed_vae:
            consumer_config = torch.zeros(
                (3), dtype=torch.int64, device=torch.cuda.current_device()
            )
            consumer_config[0] = args.iteration
            consumer_config[1] = args.consumed_train_samples
            consumer_config[2] = args.consumed_valid_samples

            from teletron.core.parallel_state import get_comm_pair
            comm_pair = get_comm_pair()

            if comm_pair is not None:
                req = dist.isend(tensor=consumer_config, dst=comm_pair.producer, tag=0)
                req.wait()
        print_datetime('after dataloaders are built')
        print_rank_0('done with setup ...')


        if not args.skip_train:
            print_rank_0('training ...')

            if args.dataloader_type == 'cyclic' and args.retro_project_dir:
                assert args.retro_cyclic_train_iters is not None
                args.train_iters = args.retro_cyclic_train_iters
                print_rank_0("retro cyclic train iters : %d" % args.train_iters)

            iteration = 0
            if args.do_train and args.train_iters > 0:
                iteration, num_floating_point_operations_so_far = self.train(
                    forward_step_func,
                    # forward_step_func,
                    self.model, self.optimizer, self.scheduler,
                    self.train_itrt, self.valid_itrt,
                    process_non_loss_data_func, self.config)

            print_datetime('after training is done')

            if args.save and iteration != 0 and iteration % args.save_interval != 0:
                self.save_checkpoint(iteration, self.model, self.optimizer, self.scheduler,
                                num_floating_point_operations_so_far)
        else:
            print_rank_0('skipping training (--skip-train is on) ...')
            iteration = args.iteration

        if args.do_valid:
            prefix = f'iteration {iteration} on validation set'
            self.evaluate_and_print_results(prefix, forward_step_func,
                                    self.valid_itrt, self.model,
                                    iteration, process_non_loss_data_func, self.config,
                                    verbose=True, write_to_tensorboard=not args.skip_train)

        if args.do_test:
            prefix = f'iteration {iteration} on test set'
            self.evaluate_and_print_results(prefix, forward_step_func,
                                    self.test_itrt, self.model,
                                    iteration, process_non_loss_data_func, self.config,
                                    verbose=True, write_to_tensorboard=not args.skip_train)


    def save_checkpoint_and_time(self, iteration, model, optimizer, opt_param_scheduler,
                             num_floating_point_operations_so_far):
        args = get_args()
        # Extra barrier is added to make sure all ranks report the max time.
        self.save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                        num_floating_point_operations_so_far)

        # if args.log_progress:
        #     compute_throughputs_and_append_to_progress_log(iteration,
        #                                                 num_floating_point_operations_so_far)



    def train(self, 
                   forward_step_func,
                   model,
                   optimizer,
                   opt_param_scheduler,
                   train_data_iterator, 
                   valid_data_iterator,
                   process_non_loss_data_func,
                   config):
        args = get_args()
        for model_module in model:
            model_module.train()
        total_loss_dict = {}

        # Iterations.
        iteration = args.iteration

        num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

        # Setup some training config params
        config.grad_scale_func = self.optimizer.scale_loss
        if isinstance(model[0], DDP) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, \
                ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
                'a custom no_sync_func is not supported when overlapping grad-reduce')
            config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
            if len(model) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if args.delay_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
                if len(model) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if args.overlap_param_gather and args.delay_param_gather:
            config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
                                    for model_index in range(len(model))]
            if len(model) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = finalize_model_grads

        print_datetime('before the start of training step')
        report_memory_flag = True
        exit = False

        if args.manual_gc:
            # Disable the default garbage collector and perform the collection manually.
            # This is to align the timing of garbage collection across ranks.
            assert args.manual_gc_interval >= 0, \
                'Manual garbage collection interval should be laerger than or equal to 0.'
            gc.disable()
            gc.collect()

        num_microbatches = get_num_microbatches()
        eval_duration = 0.0
        eval_iterations = 0
        # def track_e2e_metrics():
        #     # Nested function to track a bunch of E2E APP metrics
        #     if one_logger:
        #         train_samples = args.consumed_train_samples - train_samples_start
        #         train_iterations = iteration - iteration_start
        #         if eval_iterations:
        #             validation_iterations_time_msecs_avg = (eval_duration * 1000.0) / eval_iterations
        #         else:
        #             validation_iterations_time_msecs_avg = None

        #         one_logger.log_metrics({
        #             'train_iterations_end': iteration,
        #             'train_samples_end': args.consumed_train_samples,
        #             'train_iterations': train_iterations,
        #             'train_samples': train_samples,
        #             'validation_iterations_time_msecs_avg': validation_iterations_time_msecs_avg
        #         })

        while iteration < args.train_iters:
            if args.profile and \
            iteration == args.profile_step_start and \
            torch.distributed.get_rank() in args.profile_ranks:
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

            # Update number of microbatches first without consistency check to decide if a
            # checkpoint should be saved. If the number of microbatches is different
            # from the previous iteration, save a checkpoint. Then run consistency check
            # to make sure training configuration is still valid.
            update_num_microbatches(args.consumed_train_samples, consistency_check=False)
            if get_num_microbatches() != num_microbatches and iteration != 0:
                assert get_num_microbatches() > num_microbatches, \
                    "number of microbatches should be increasing due to batch size rampup"
                self.save_checkpoint_and_time(iteration, model, optimizer,
                                        opt_param_scheduler,
                                        num_floating_point_operations_so_far)
            num_microbatches = get_num_microbatches()
            update_num_microbatches(args.consumed_train_samples, consistency_check=True)

            args.curr_iteration = iteration
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                self.train_step(forward_step_func,
                        train_data_iterator,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        config)
            iteration += 1
            batch_size = mpu.get_data_parallel_world_size() * \
                        args.micro_batch_size * \
                        get_num_microbatches()
            args.consumed_train_samples += batch_size
            num_floating_point_operations_so_far += num_floating_point_operations(args, batch_size)

            # Logging.
            # loss_scale = optimizer.get_loss_scale().item()
            # params_norm = None
            # if args.log_params_norm:
            #     params_norm = calc_params_l2_norm(model)

            # # if iteration % args.log_interval == 0:
            # #     track_e2e_metrics()

            # learning_rate = None
            # decoupled_learning_rate = None
            # for param_group in optimizer.param_groups:
            #     if param_group['is_decoupled_lr']:
            #         decoupled_learning_rate = param_group['lr']
            #     else:
            #         learning_rate = param_group['lr']
            # report_memory_flag = training_log(loss_dict, total_loss_dict,
            #                                 learning_rate,
            #                                 decoupled_learning_rate,
            #                                 iteration, loss_scale,
            #                                 report_memory_flag, skipped_iter,
            #                                 grad_norm, params_norm, num_zeros_in_grad)

            # Autoresume
            # if args.adlr_autoresume and \
            # (iteration % args.adlr_autoresume_interval == 0):
            #     check_adlr_autoresume_termination(iteration, model, optimizer,
            #                                     opt_param_scheduler)

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0 and \
                    args.do_valid:
                if args.use_distributed_optimizer and args.overlap_param_gather:
                    optimizer.disable_pre_hook()
                if args.manual_gc and args.manual_gc_eval:
                    # Collect all objects.
                    gc.collect()
                prefix = 'iteration {}'.format(iteration)
                self.evaluate_and_print_results(prefix, forward_step_func,
                                        valid_data_iterator, model,
                                        iteration, process_non_loss_data_func,
                                        config, False)
                eval_iterations += args.eval_iters
                if args.manual_gc and args.manual_gc_eval:
                    # Collect only the objects created and used in evaluation.
                    gc.collect(generation=0)
                if args.use_distributed_optimizer and args.overlap_param_gather:
                    optimizer.enable_pre_hook()

            # Checkpointing
            saved_checkpoint = False
            if args.save and args.save_interval and \
                            iteration % args.save_interval == 0:
                self.save_checkpoint_and_time(iteration, model, optimizer,
                                        opt_param_scheduler,
                                        num_floating_point_operations_so_far)
                saved_checkpoint = True

            # Exiting based on duration
            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins],
                    dtype=torch.int, device='cuda')
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    if not saved_checkpoint:
                        self.save_checkpoint_and_time(iteration, model, optimizer,
                                                opt_param_scheduler,
                                                num_floating_point_operations_so_far)
                    print_datetime('exiting program after {} minutes'.format(train_time))
                    exit = True
                    break

            # Exiting based on iterations
            if args.exit_interval and iteration % args.exit_interval == 0:
                if args.save and not saved_checkpoint:
                    self.save_checkpoint_and_time(iteration, model, optimizer,
                                            opt_param_scheduler,
                                            num_floating_point_operations_so_far)
                torch.distributed.barrier()
                print_datetime('exiting program at iteration {}'.format(iteration))
                exit = True
                break

            if args.profile and \
            iteration == args.profile_step_end and \
            torch.distributed.get_rank() in args.profile_ranks:
                torch.cuda.cudart().cudaProfilerStop()

            if args.manual_gc:
                if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                    gc.collect()

        # track_e2e_metrics()

        # Flush TensorBoard and WandB writers.
        # writer = get_tensorboard_writer()
        # if writer:
        #     writer.flush()
        # wandb_writer = get_wandb_writer()
        # if wandb_writer:
        #     wandb_writer.finish()

        # Close out pre-hooks if using distributed optimizer and overlapped param gather.
        if args.use_distributed_optimizer and args.overlap_param_gather:
            optimizer.disable_pre_hook()

        # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
        if exit:
            sys.exit()

        return iteration, num_floating_point_operations_so_far
    
    def train_step(self, forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
        """Single training step."""
        args = get_args()

        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # import os, debugpy
        # dist.barrier()  
        # if int(os.environ.get("RANK","0")) == 0:
        #     debugpy.breakpoint()
        # dist.barrier()
        # Forward pass.
        forward_backward_func = get_forward_backward_func()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)

        # breakpoint()
        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Vision gradients.
        if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(model[0])
            unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Vision momentum.
        if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(model[0])
            unwrapped_model.update_momentum(args.curr_iteration)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return {}, skipped_iter, grad_norm, num_zeros_in_grad

    def evaluate_and_print_results(self,
                                   prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func, config,
                               verbose=False, write_to_tensorboard=True):
        """Helper function to evaluate and dump results on screen."""
        args = get_args()

        total_loss_dict, collected_non_loss_data, timelimit = self.evaluate(
            forward_step_func, data_iterator, model,
            process_non_loss_data_func, config, verbose)
        # Timelimit hit during evaluation
        if timelimit:
            return
        string = ' validation loss at {} | '.format(prefix)
        import math
        for key in total_loss_dict:
            string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
            ppl = math.exp(min(20, total_loss_dict[key].item()))
            string += '{} PPL: {:.6E} | '.format(key, ppl)

        length = len(string) + 1
        print_rank_last('-' * length)
        print_rank_last(string)
        print_rank_last('-' * length)

    def evaluate(self,
                forward_step_func,
                data_iterator,
                model,
                process_non_loss_data_func,
                config,
                verbose=False):
        """Evaluation."""
        args = get_args()

        # if args.vision_pretraining and args.vision_pretraining_type == "dino":
        #     from megatron.legacy.model.vision.knn_monitor import compute_feature_bank
        #     compute_feature_bank(model)

        # Turn on evaluation mode which disables dropout.
        for model_module in model:
            model_module.eval()

        total_loss_dict = {}

        # make validation batch size independent from training batch size
        eval_batch_size = args.global_batch_size
        eval_num_microbatches = eval_batch_size // \
            (args.micro_batch_size * args.data_parallel_size)

        with torch.no_grad():
            iteration = 0
            if verbose:
                print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
            while iteration < args.eval_iters:
                iteration += 1
                if verbose:
                    print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

                forward_backward_func = get_forward_backward_func()
                # Don't care about timing during evaluation
                config.timers = None
                loss_dicts = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=eval_num_microbatches,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True)

                # Empty unused memory
                if args.empty_unused_memory_level >= 1:
                    torch.cuda.empty_cache()

                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    # Reduce across processes.
                    for loss_dict in loss_dicts:
                        for key in loss_dict:
                            total_loss_dict[key] = total_loss_dict.get(
                                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]

                args.consumed_valid_samples += eval_batch_size

                if args.exit_duration_in_mins:
                    train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                    done_cuda = torch.tensor(
                        [train_time > args.exit_duration_in_mins],
                        dtype=torch.int, device='cuda')
                    torch.distributed.all_reduce(
                        done_cuda, op=torch.distributed.ReduceOp.MAX)
                    done = done_cuda.item()
                    if done:
                        print_rank_0('Exiting during evaluation, timelimit reached')
                        return None, None, True

            collected_non_loss_data = None
            if process_non_loss_data_func is not None and is_last_rank():
                collected_non_loss_data = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=get_num_microbatches(),
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True,
                    collect_non_loss_data=True)

        # Move model back to the train mode.
        for model_module in model:
            model_module.train()

        for key in total_loss_dict:
            total_loss_dict[key] /= args.eval_iters * eval_num_microbatches

        return total_loss_dict, collected_non_loss_data, False
