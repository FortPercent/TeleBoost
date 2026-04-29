from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.sharding_manager.diffusion import DiffusionBaseShardingManager

from .dancegrpo_fsdp_worker import DiffusionActorRolloutRefWorker
from .diffusion_rollout_pixel import DiffusionRolloutPixel
from .dp_actor_pixel import DiffusionDataParallelPPOActorPixel

__all__ = ["DiffusionActorRolloutRefWorkerPixel"]


class DiffusionActorRolloutRefWorkerPixel(DiffusionActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            self.actor = DiffusionDataParallelPPOActorPixel(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )
        if self._is_ref:
            self.ref_policy = DiffusionDataParallelPPOActorPixel(
                config=self.config.ref,
                actor_module=self.ref_module_fsdp,
            )

    def _build_rollout(self, trust_remote_code=False):
        if self.config.type == "diffusion":
            rollout = DiffusionRolloutPixel(module=self.actor_module_fsdp, config=self.config)
            rollout_sharding_manager = DiffusionBaseShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=None,
                model_config=self.actor_model_config,
                offload_param=self._is_offload_param,
            )
            return rollout, rollout_sharding_manager
        return super()._build_rollout(trust_remote_code)
