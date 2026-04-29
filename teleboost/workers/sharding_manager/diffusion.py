# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os
import time
from collections import OrderedDict

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from dataclasses import asdict

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fsdp_utils import fsdp_version, layered_summon_lora_params, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.model import convert_weight_keys
from verl.utils.torch_functional import check_device_is_available
from verl.utils.vllm_utils import TensorLoRARequest, VLLMHijack, is_version_ge, patch_vllm_moe_model_weight_loader

from verl.workers.sharding_manager.base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

from verl import DataProto


class DiffusionBaseShardingManager:
    @check_device_is_available()
    def __init__(self, module: FSDP, inference_engine: LLM, model_config, full_params: bool = False, device_mesh: DeviceMesh = None, offload_param: bool = False, load_format: str = "dummy_hf", layered_summon: bool = True):
        self.module = module        
        self.offload_param = offload_param
        # For AsyncLLM, inference_engine and model_runner are defer initialized in vLLMAsyncRollout.load_model
        # self.inference_engine = inference_engine
        # self.model_runner = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if inference_engine else None

        # if "vllm_v_0_6_3" in str(type(self.inference_engine)) or "vllm_v_0_5_4" in str(type(self.inference_engine)):
        #     # vLLM <= v0.6.3
        #     self.model_runner = self.inference_engine.llm_engine.model_executor.worker.model_runner if self.inference_engine else None
        # else:
        #     # vLLM > v0.6.3
        #     self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None

        # self.model_config = model_config
        # self.device_mesh = device_mesh

        # self.load_format = load_format
        # self.layered_summon = layered_summon

        # # Full params
        # self.full_params = full_params
        # if full_params and fsdp_version(self.module) == 1:
        #     FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        # elif fsdp_version(self.module) == 1:
        #     FSDP.set_state_dict_type(
        #         self.module,
        #         state_dict_type=StateDictType.SHARDED_STATE_DICT,
        #         state_dict_config=ShardedStateDictConfig(),
        #     )

        # self.tp_size = self.device_mesh["infer_tp"].size()
        # self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # # Note that torch_random_states may be different on each dp rank
        # self.torch_random_states = get_torch_device().get_rng_state()
        # # get a random rng states
        # if self.device_mesh is not None:
        #     gen_dp_rank = self.device_mesh["dp"].get_local_rank()
        #     get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        #     self.gen_random_states = get_torch_device().get_rng_state()
        #     get_torch_device().set_rng_state(self.torch_random_states)
        # else:
        #     self.gen_random_states = None

        # self.base_sync_done: bool = "dummy" not in load_format
        # if is_version_ge(pkg="vllm", minver="0.7.3"):
        #     VLLMHijack.hijack()
            
    def __enter__(self):
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)
            
        # add empty cache after each compute
        get_torch_device().empty_cache()

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data