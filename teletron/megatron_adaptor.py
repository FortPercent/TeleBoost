# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.
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

import megatron.training
import megatron.core
from teletron.core.parallel_state import initialize_model_parallel_decorators
from teletron.core.training import setup_model_and_optimizer_decorators

def exe_adaptation():
    megatron.core.parallel_state.initialize_model_parallel = initialize_model_parallel_decorators(
        megatron.core.parallel_state.initialize_model_parallel
    )
    megatron.core.mpu = megatron.core.parallel_state

    megatron.training.training.setup_model_and_optimizer = setup_model_and_optimizer_decorators(
        megatron.training.training.setup_model_and_optimizer
    )

exe_adaptation()