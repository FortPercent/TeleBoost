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

### Option 1: Using Reward Models (`reward_model.enable: true`)

#### Reward Model Types

The system supports three types of reward models, configured in `recipe/dancegrpo/dancegrpo_fsdp_worker.py`:

---

##### 🎯 **Type 1: Qwen**

**Class:** `QwenRewardModelWorker(RewardModelWorker)`

Uses vLLM for distributed inference with Qwen VL model for video quality evaluation.

**Configuration Parameters:**

```yaml
reward_model:
  enable: true
  strategy: diffusion
  type: qwen  # or model_name: qwen
  micro_batch_size_per_gpu: 1
  
  # Model path
  model:
    path: 'path/to/Qwen2.5-VL-7B-Instruct'
    use_shm: true  # Use shared memory for faster loading
  
  # vLLM rollout configuration
  rollout:
    name: vllm
    load_format: safetensors  # ⚠️ Required for Qwen models
    tensor_model_parallel_size: 1
    
    # Sampling parameters (configurable)
    temperature: 0.8  # default: 0.8
    top_p: 0.9        # default: 0.9
    max_tokens: 128   # default: 128
  
  # Video processing configuration
  extra_config:
    max_pixels: 151200  # default: 360*420
    fps: 1.0            # default: 1.0
    video_base_path: /path/to/videos/  # Optional, replaces "./" in paths

# Training parameters
actor_rollout_ref:
  rollout:
    n: 4
    temperature: 1.0
    top_p: 0.9

vllm_mode: spmd
world_size: 8
```

---

> **Note**: Qwen models use a specialized `QwenRewardModelWorker` with vLLM for distributed inference. If you set `model_name: qwen` or `type: qwen`, the system will automatically use the vLLM-based worker.

##### 🎯 **Type 2: Single (Unified Registry-Based)**

Use a single reward model from the Registry. Simply specify `model_name` to automatically load any registered model.

**Configuration:**

```yaml
reward_model:
  enable: true
  strategy: diffusion
  type: single
  micro_batch_size_per_gpu: 1      # ⚠️ Mandatory field
  
  # Specify any registered model name
  # Available models: aesthetic, hps, raft, videoclip, videophy, qwen
  model_name: hps
  
  # Model weights path
  model:
    path: /path/to/HPS_v2.1_compressed.pt
  
  # Model-specific configuration
  extra_config:
    model_type: ViT-H-14  # HPS-specific option
  
  normalize: true
```

**Example: Using Aesthetic Model**

```yaml
reward_model:
  type: single
  model_name: aesthetic
  extra_config:
    clip_model_path: /path/to/ViT-L-14.pt
    aes_model_path: /path/to/sa_0_4_vit_l_14_linear.pth
```

##### 🎯 **Type 3: Joint (Dynamic & Extensible)**

The `joint` strategy allows you to combine multiple reward models with flexible weighting, normalization, and GPU resource allocation. Each model runs as an independent worker group, enabling true parallel execution.

**Configuration Structure:**

```yaml
reward_model:
  enable: true
  strategy: diffusion
  type: joint
  micro_batch_size_per_gpu: 1      # ⚠️ Mandatory field

  joint:
    # Aggregation method: weighted_sum (default), mean, max, min
    aggregation: weighted_sum
    
    # Normalization settings
    normalize_individual: true     # Normalize each model's score before aggregation
    normalize_final: false         # Normalize the final aggregated score
    
    # Dict format: model_name -> config (recommended)
    models:
      aesthetic:
        enabled: true
        weight: 1.0
        mps_percentage: 30         # GPU resource allocation via MPS
        extra_config:
          clip_model_path: /path/to/ViT-L-14.pt
          aes_model_path: /path/to/sa_0_4_vit_l_14_linear.pth
          
      raft:
        enabled: true
        weight: 1.0
        mps_percentage: 30
        extra_config:
          raft_model_path: /path/to/raft-things.pth
          
      videoclip:
        enabled: true
        weight: 1.5
        mps_percentage: 25
        extra_config:
          model_path: /path/to/VideoCLIP-XL.bin
          
      videophy:
        enabled: true
        weight: 0.8
        mps_percentage: 25
        extra_config:
          model_path: /path/to/videocon_physics
```

**Note:** Each model runs in its own worker group with the specified MPS percentage for GPU resource sharing. This enables true parallel execution across models.

#### Adding Custom Reward Models (New Registry System)

The new system allows adding reward models **without modifying core code**.

1. **Create a file** in `recipe/dancegrpo/reward_models/` (e.g., `my_model.py`).
2. **Implement & Register** your class:

```python
from .base import BaseRewardModel, RewardConfig
from .registry import RewardRegistry
import torch

@RewardRegistry.register("my_awesome_model")  # Register with a unique name
class MyAwesomeRewardModel(BaseRewardModel):
    
    REWARD_KEY = "my_awesome_rewards"  # Output key in the result batch
    
    def init_model(self) -> None:
        """Initialize and load the model weights."""
        if not self.is_active:
            return
        
        # Access config.extra_config for custom parameters
        model_path = self.config.extra_config.get("model_path", "")
        self.model = load_your_model(model_path)
        self.move_model_to_device(self.model)
        
    def compute_single_score(self, video_frames: torch.Tensor, caption: str) -> float:
        """Compute reward for a single sample.
        
        Args:
            video_frames: [T, C, H, W] tensor of video frames
            caption: The text caption/prompt
            
        Returns:
            A float score
        """
        with torch.no_grad():
            score = self.model(video_frames, caption)
        return score.item()
```

3. **Import it** in `recipe/dancegrpo/reward_models/__init__.py`:

```python
from .my_model import MyAwesomeRewardModel
```

4. **Configure** it in `dancegrpo_trainer.yaml`:

```yaml
reward_model:
  type: joint
  joint:
    models:
      my_awesome_model:
        enabled: true
        weight: 2.0
        mps_percentage: 25
        extra_config:
          model_path: /path/to/model
          my_param: "value"
```


#### Reward Managers

**Location:** `verl/workers/reward_manager`

**Available Managers:**
- `BatchRewardManager`
- `DAPORewardManager`
- `NaiveRewardManager`
- `PrimeRewardManager`
- `AIGCRewardManager` - Registered as `'dancegrpo'`

To add a custom reward manager, create a new Python file in `verl/workers/reward_manager` and register it using the `@register()` decorator.

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