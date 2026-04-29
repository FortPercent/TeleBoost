import logging

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

from .dancegrpo_ray_trainer_pixel import RayDanceGRPOTrainerPixel
from .main_dancegrpo import (
    _build_reward_fns,
    _build_tokenizer_and_processor,
    _init_ray,
    _register_reward_workers,
    _validate_config,
)

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class TaskRunnerPixel:
    def run(self, config: DictConfig) -> None:
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils.dataset.rl_dataset import wan_preprocessed_collate_function
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        _validate_config(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        tokenizer, processor = _build_tokenizer_and_processor(local_path, config.actor_rollout_ref.model)
        ray_worker_group_cls, actor_rollout_worker_cls = _resolve_actor_worker_and_group_pixel(config)

        role_worker_mapping = {}
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {}

        def register_role(role, worker_cls):
            role_worker_mapping[role] = ray.remote(worker_cls)
            mapping[role] = global_pool_id

        register_role(Role.ActorRollout, actor_rollout_worker_cls)
        _register_reward_workers(config, role_worker_mapping, mapping, global_pool_id)
        reward_fn, val_reward_fn = _build_reward_fns(config, tokenizer)

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        trainer = RayDanceGRPOTrainerPixel(
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
        trainer.init_workers()
        trainer.fit()


def _resolve_actor_worker_and_group_pixel(config: DictConfig):
    strategy = config.actor_rollout_ref.actor.strategy
    use_critic = config.algorithm.get("adv_estimator", "grpo") == "gae"
    if use_critic and hasattr(config, "critic"):
        assert strategy == config.critic.strategy

    if strategy == "fsdp":
        from verl.single_controller.ray import RayWorkerGroup
        from .dancegrpo_fsdp_worker_pixel import DiffusionActorRolloutRefWorkerPixel

        return RayWorkerGroup, DiffusionActorRolloutRefWorkerPixel

    from .main_dancegrpo import _resolve_actor_worker_and_group

    return _resolve_actor_worker_and_group(config)


def run_ppo(config: DictConfig) -> None:
    _init_ray(config)

    runner = TaskRunnerPixel.remote()
    ray.get(runner.run.remote(config))


@hydra.main(config_path="config", config_name="dancegrpo_trainer_pixel", version_base=None)
def main(config: DictConfig) -> None:
    run_ppo(config)


if __name__ == "__main__":
    main()
