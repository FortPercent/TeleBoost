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

```yaml
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

##### 🔹 qwen

**Class:** `QwenRewardModelWorker(RewardModelWorker)`

**Configuration Parameters:**

```yaml
# Model Configuration
reward_model:
  model:
    path: 'path/to/Qwen2.5-VL-7B-Instruct'
  rollout:
    load_format: safetensors  # ⚠️ Required for Qwen models

# Training Parameters
actor_rollout_ref:
  rollout:
    n: 4                                      # Number of samples per prompt
    temperature: 1.0
    top_p: 0.9
    tensor_model_parallel_size: 1             # Tensor parallelism for rollout

algorithm:
  adv_estimator: grpo                         # Advantage estimator

trainer:
  type: diffusion

# Advanced Settings
rollout:
  name: vllm
  mode: sync
  layered_summon: false

use_rm: true
vllm_mode: spmd
world_size: 8                                 # Number of GPUs
use_shm: true
```

---

##### 🔹 single

**Class:** `DiffusionRewardModelWorker(RewardModelWorker)`

**Configuration:**

```yaml
reward_model:
  model:
    path: 'path/to/HPSv2/HPS_v2_compressed.pt'
```

---

##### 🔹 joint

**Classes:**
- `AestheticRewardModelWorker(RewardModelWorker)`
- `RAFTRewardModelWorker(RewardModelWorker)`
- `VideoclipRewardModelWorker(RewardModelWorker)`
- `VideophyRewardModelWorker(RewardModelWorker)`
- `MultiRewardModelWorker(RewardModelWorker)` - Combines all four models

**Configuration:**

```yaml
reward_model:
  type: joint
  aesthetic:
    clip_model_path: /path/to/ViT-L-14.pt
    aes_model_path: /path/to/sa_0_4_vit_l_14_linear.pth
  raft:
    model_path: /path/to/raft-things.pth
  videoclip:
    model_path: /path/to/VideoCLIP-XL.bin
  videophy:
    model_path: /path/to/model/arena_model/videocon_physics
```

#### Adding Custom Reward Models

1. Create a new class inheriting from `RewardModelWorker` in `dancegrpo_fsdp_worker.py`
2. Register the worker:

```python
from .dancegrpo_fsdp_worker import NewRewardModelWorker as RewardModelWorker
role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
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

### Option 2: Custom Reward Functions (`reward_model.enable: false`)

When `use_rm: false`, you can use custom reward functions:

#### Method 1: Custom Reward Function

Define a custom reward function in `verl/trainer/config/ppo_trainer.yaml`:

```yaml
custom_reward_function:
  path: path/to/your_script.py      # Path to your Python script
  name: compute_score
```

The system will automatically register this function and use it as `compute_score`.

#### Method 2: Default Reward Function

If no custom function is defined, the system uses `default_compute_score` from `verl/trainer/ppo/reward.py`.