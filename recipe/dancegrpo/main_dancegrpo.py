import hydra

# os.environ["RAY_DEDUP_LOGS"] = "0"
import ray

from verl.trainer.ppo.reward import get_custom_reward_fn

from .dancegrpo_ray_trainer import RayDanceGRPOTrainer

RAY_ENV_VARS = {
    "TOKENIZERS_PARALLELISM": "true",
    "NCCL_DEBUG": "WARN",
    "VLLM_LOGGING_LEVEL": "WARN",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def _init_ray(config) -> None:
    if ray.is_initialized():
        return
    # this is for local ray cluster
    ray.init(
        runtime_env={"env_vars": RAY_ENV_VARS},
        num_cpus=config.ray_init.num_cpus,
    )


def _build_tokenizer_and_processor(local_path):
    import os
    from verl.utils import hf_processor, hf_tokenizer

    tokenizer_path = os.path.join(local_path, "google/umt5-xxl")
    tokenizer = hf_tokenizer(tokenizer_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none
    return tokenizer, processor


def _resolve_actor_worker_and_group(config):
    strategy = config.actor_rollout_ref.actor.strategy
    assert strategy == config.critic.strategy

    if strategy == "fsdp":  # actor的策略
        from verl.single_controller.ray import RayWorkerGroup
        from .dancegrpo_fsdp_worker import DiffusionActorRolloutRefWorker

        return RayWorkerGroup, DiffusionActorRolloutRefWorker

    if strategy == "megatron":
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker

        return NVMegatronRayWorkerGroup, ActorRolloutRefWorker

    raise NotImplementedError(f"Unknown actor strategy: {strategy}")


def _build_reward_fns(config, tokenizer):
    from verl.workers.reward_manager import get_reward_manager_cls

    # Note(haibin.lin): please make sure custom reward managers are imported and
    # registered via `verl.workers.reward_manager.register`
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)  # 这里看注册的reward manager是什么
    compute_score = get_custom_reward_fn(config)  # 这里返回None，表示没有自定义的reward function

    # 初始化reward function，这个函数会被传入每一个worker中，作为计算reward的接口
    # 使用的reward manager 是 dancegrpo(从config中读取的) -> AIGCRewardManager
    reward_fn = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=compute_score,  # 没有定义的话这里会自动转为默认的default_compute_score函数
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=1,
        compute_score=compute_score,
    )
    return reward_fn, val_reward_fn


def run_ppo(config) -> None:
    _init_ray(config)

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@hydra.main(config_path="config", config_name="dancegrpo_trainer", version_base=None)
def main(config):
    run_ppo(config)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        # 需要tokenizer和preprocessor
        tokenizer, processor = _build_tokenizer_and_processor(local_path)

        # define worker classes
        ray_worker_group_cls, actor_rollout_worker_cls = _resolve_actor_worker_and_group(config)

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Dict[Role, RayClass]
        role_worker_mapping = {}

        global_pool_id = "global_pool"
        # Dict[str, List[int]]
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # Dict[Role, str]
        # global_pool_id 本身也是一个Dict[str, List[int]]
        mapping = {}

        def register_role(role, worker_cls):
            # ray.remote(Worker).remote() 这才是启动一个远程worker
            role_worker_mapping[role] = ray.remote(worker_cls)
            mapping[role] = global_pool_id

        # 调用ActorRollout，需要通过ray.remote的方式实例化worker
        register_role(Role.ActorRollout, actor_rollout_worker_cls)

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data

        if config.reward_model.enable:  # reward model的策略
            # 以何种分布式并行方式加载这个奖励模型 LLM（大语言模型）
            if config.reward_model.strategy == "fsdp":
                from .fsdp_worker import RewardModelWorker
                register_role(Role.RewardModel, RewardModelWorker)
                print(f"Mapping type {Role.RewardModel} to be the reward model")

            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
                register_role(Role.RewardModel, RewardModelWorker)
                print(f"Mapping type {Role.RewardModel} to be the reward model")

            # reward model 使用 diffusion 模型
            elif config.reward_model.strategy == "diffusion":  # reward使用的是diffusion
                if config.reward_model.type == "qwen":  # reward model的类型
                    from .dancegrpo_fsdp_worker import QwenRewardModelWorker as RewardModelWorker
                    register_role(Role.RewardModel, RewardModelWorker)
                    print(f"Mapping type {Role.RewardModel} to be the Qwen reward model")

                elif config.reward_model.type == "single":
                    from .dancegrpo_fsdp_worker import DiffusionRewardModelWorker as RewardModelWorker
                    register_role(Role.RewardModel, RewardModelWorker)
                    print(f"Mapping type {Role.RewardModel} to be the single diffusion reward model")

                elif config.reward_model.type == "joint":
                    from .dancegrpo_fsdp_worker import (
                        AestheticRewardModelWorker,
                        RAFTRewardModelWorker,
                        VideoclipRewardModelWorker,
                        VideophyRewardModelWorker,
                    )
                    # 注意：joint 模式下可能不需要单一的 RewardModelWorker
                    # 而是注册多个具体角色的 worker
                    register_role(Role.AestheticRewardModel, AestheticRewardModelWorker)
                    register_role(Role.RAFTRewardModel, RAFTRewardModelWorker)
                    register_role(Role.VideoclipRewardModel, VideoclipRewardModelWorker)
                    register_role(Role.VideophyRewardModel, VideophyRewardModelWorker)

                    print("Mapping multiple reward models for 'joint' type", role_worker_mapping)

                else:
                    raise NotImplementedError(f"Unknown diffusion reward model type: {config.reward_model.type}")

            else:
                raise NotImplementedError(f"Unknown reward model strategy: {config.reward_model.strategy}")

        reward_fn, val_reward_fn = _build_reward_fns(config, tokenizer)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import wan_preprocessed_collate_function

        trainer = RayDanceGRPOTrainer(  # 初始化训练类
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            collate_fn=wan_preprocessed_collate_function,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()  # 初始化所有的worker
        trainer.fit()  # 开始训练


if __name__ == "__main__":
    main()
