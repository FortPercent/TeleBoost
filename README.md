# DanceGRPO

A reinforcement learning framework for diffusion models using Group Relative Policy Optimization (GRPO).

## 📖 Table of Contents

- [Setup](#-setup)
- [Data Preprocessing](#-data-preprocessing)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Reward Models](#-reward-models)

---

## 🛠 Setup

```bash
# Clone repository
git clone https://qiruoling@code.srdcloud.cn/P24HQASYF0004/AI-Infra/Dance-grpo
cd Dance-grpo

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Data Preprocessing

Preprocess your data before training:

```bash
bash data_preprocess/preprocess_wan_rl_embeddings.sh
```

---

## 🏃 Usage

### Quick Start

```bash
bash recipe/dancegrpo/run_dancegrpo.sh
```

### Main Scripts

The main training scripts are organized as follows:

```
recipe/dancegrpo/
├── main_dancegrpo.py              # Main entry point
├── dancegrpo_fsdp_worker.py       # FSDP worker implementation
├── dancegrpo_ray_trainer.py       # Ray trainer implementation
├── dp_actor.py                    # Data parallel actor
└── run_dancegrpo.sh               # Execution script
```

---

## ⚙️ Configuration

### Configuration Files

```
├── verl/trainer/config/ppo_trainer.yaml                  # Base PPO configuration
├── recipe/dancegrpo/config/dancegrpo_trainer.yaml        # DanceGRPO configuration
└── recipe/dancegrpo/run_dancegrpo.sh                     # Runtime parameters
```

### Key Configuration Parameters

In `dancegrpo_trainer.yaml`:

```
reward_model:
  enable: true                    # Enable/disable reward model
  strategy: diffusion             # Reward strategy
  type: qwen/single/joint         # Reward model type
  
reward_manager: dancegrpo         # Reward manager type
```

---

## 🎯 Reward Models

The framework uses a **Unified Reward Architecture** that supports single models, joint models (parallel execution), and custom registry-based models.

### Architecture Overview

- **UnifiedRewardModelWorker**: The core worker class for all registry-based models. It inherits from `Worker` and `WorkerProfilerExtension`.
- **JointRewardModelWorker**: Orchestrates multiple reward models in joint mode.
- **Independent Worker Groups**: In joint mode, each reward model runs in its own Ray worker group, allowing for:
  - **True Parallel Execution**: Models run concurrently.
  - **Independent Resource Allocation**: Each model can have its own `mps_percentage` (e.g., 30% for Model A, 20% for Model B).
  - **Isolated Configuration**: Each model has its own configuration without conflicts.

### Option 1: Using Reward Models (`reward_model.enable: true`)

#### 🎯 **Type 1: Qwen (vLLM)**

**Class:** `QwenRewardModelWorker`

Uses vLLM for distributed inference with Qwen VL model. Best for complex visual reasoning.

**Configuration:**
```yaml
reward_model:
  enable: true
  type: qwen
  micro_batch_size_per_gpu: 1
  model:
    path: /path/to/Qwen2.5-VL
  rollout:
    name: vllm
    tensor_model_parallel_size: 1
```

#### 🎯 **Type 2: Single (Unified Registry-Based)**

**Class:** `UnifiedRewardModelWorker`

Use any single reward model from the Registry.

**Configuration:**
```yaml
reward_model:
  type: single
  model_name: aesthetic  # Specify registered model name
  micro_batch_size_per_gpu: 1
  
  # Model-specific config (differs by model)
  extra_config:
    clip_model_path: /path/to/clip
    aes_model_path: /path/to/aesthetic
```

#### 🎯 **Type 3: Joint (Parallel & Dynamic)**

**Class:** `JointRewardModelWorker` (Coordinator) & `UnifiedRewardModelWorker` (Individual Models)

Combines multiple reward models. **Constraint:** Due to Ray limitations, independent worker groups are used to support per-model configuration and MPS.

**Configuration:**
```yaml
reward_model:
  type: joint
  micro_batch_size_per_gpu: 1

  joint:
    aggregation: weighted_sum  # weighted_sum, mean, min, max
    
    # Dict of models (Key = Registered Name)
    models:
      aesthetic:
        enabled: true
        weight: 1.0
        mps_percentage: 20      # Allocates 20% of GPU compute
        extra_config:
          clip_model_path: ...
          aes_model_path: ...
          
      raft:
        enabled: true
        weight: 1.5
        mps_percentage: 30
        model_path: /path/to/raft  # Can also specify path here
        extra_config:
          stride: 1
          
      videophy:
        enabled: true
        weight: 0.8
        mps_percentage: 25
        model_path: /path/to/videophy
```

### 🛠️ Adding Custom Reward Models

The system is designed for extensibility via the `RewardRegistry`.

1. **Create** a new file in `recipe/dancegrpo/reward_models/` (e.g., `my_model.py`).
2. **Implement** your class using the registry decorator:

```python
from .base import BaseRewardModel, RewardConfig
from .registry import RewardRegistry
import torch

@RewardRegistry.register("my_model")
class MyRewardModel(BaseRewardModel):
    
    REWARD_KEY = "my_model_rewards"  # Unique key for output

    def init_model(self) -> None:
        if not self.is_active: return
        
        # Access config
        model_path = self.config.model_path
        extra_param = self.config.extra_config.get("my_param", 10)
        
        # Load your model
        self.model = MyModel(model_path, extra_param)
        self.model.eval()
        self.move_model_to_device(self.model)

    def compute_single_score(self, video_frames: torch.Tensor, caption: str) -> float:
        """
        Args:
            video_frames: (T, C, H, W) tensor
            caption: Text string
        """
        # Data preparation (T, C, H, W) -> (B, C, T, H, W) if needed
        # Inference...
        return 0.95
```

3. **Register** it in `recipe/dancegrpo/reward_models/__init__.py`:
```python
from .my_model import MyRewardModel
```

4. **Use** it in `dancegrpo_trainer.yaml`:
```yaml
reward_model:
  type: joint
  joint:
    models:
      my_model:
        weight: 1.0
        model_path: /path/to/weights
```

### 🧠 Reward Managers

**Location:** `verl/workers/reward_manager`

Reward managers handle the lifecycle of reward computation.
- **`AIGCRewardManager`**: The default for DanceGRPO (registered as `'dancegrpo'`). It handles the interaction between the trainer and the Unified/Joint reward workers.

To add a custom manager, implement it in `verl/workers/reward_manager` and register it.

---

## 🧠 Algorithm Variants

Dance-GRPO supports different algorithmic strategies to optimize the diffusion training process.

### 1. Standard GRPO (Full Trajectory)

Standard GRPO performs policy optimization on the entire denoising trajectory. To use this mode, disable the flow optimization.

**Configuration:**

```yaml
flow_grpo:
  enable: false             # Disable flow optimization (Standard GRPO)
```

### 2. Flow-GRPO (Default - Optimized)

Flow-GRPO adapts the GRPO algorithm for flow matching and diffusion models by optimizing a specific time window or subsampling steps.

**Configuration (`dancegrpo_trainer.yaml`):**

```yaml
# Flow-GRPO settings
flow_grpo:
  enable: true
  sde_window_size: 2        # Window size for SDE optimization
  # Range of timesteps to optimize (e.g., [0, 16])
  sde_window_range: [0, "${actor_rollout_ref.sampling_steps}"]
  shuffle_timesteps: false  # Whether to shuffle timesteps during training
```

### 2. GRPO-Guard (Stability)

GRPO-Guard adds stability mechanisms to preventing policy collapse and exploding gradients during training.

**Configuration (`dancegrpo_trainer.yaml`):**

```yaml
actor_rollout_ref:
  actor:
    grpo_guard:
      enable: true           # Enable guard mechanism
      ratio_norm: false      # Normalize importance ratios
      grad_reweight: false   # Reweight gradients based on advantage
      ratio_norm_eps: 1e-6
      grad_reweight_eps: 1e-6
      grad_reweight_alpha: 1.0
```

---

### Option 2: Custom Reward Functions (`reward_model.enable: false`)

When `use_rm: false`, you can use custom reward functions:

#### Method 1: Custom Reward Function

Define a custom reward function in `verl/trainer/config/ppo_trainer.yaml`:

```
custom_reward_function:
  path: path/to/your_script.py      # Path to your Python script
  name: compute_score
```

The system will automatically register this function and use it as `compute_score`.

#### Method 2: Default Reward Function

If no custom function is defined, the system uses `default_compute_score` from `verl/trainer/ppo/reward.py`.