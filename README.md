# DanceGRPO

## 📖 Guide

1. [Setup](#-setup)
2. [Data Prepare](#-data-preprocess)
3. [Usage](#-usage)
4. [TODO](#-todo)
5. [Cite](#-citation)

---

## 🛠 Setup

```bash
    # clone repository
    git clone https://qiruoling@code.srdcloud.cn/P24HQASYF0004/AI-Infra/Dance-grpo
    cd Dance-grpo

    # install dependency
    pip install -r requirements.txt

```

## 📊 Data Preprocess

```bash
    bash data_preprocess\preprocess_wan_rl_embeddings.sh
```

---

## 🏃 Usage

```bash 
    bash recipe\dancegrpo\run_dancegrpo.sh
```

### 1. main script
```
recipe/dancegrpo/
├── main_dancegrpo.py         # main file
├── dancegrpo_fsdp_worker.py         # FSDP works
├── dancegrpo_ray_trainer.py         # Ray trainer
├── dp_actor.py         # dp Actor
└── run_dancegrpo.sh         # entry
```
```
conifg files：
├── verl/trainer/config/ppo_trainer.yaml     # basic PPO
├── recipe/dancegrpo/config/dancegrpo_trainer.yaml     # implementation
└── recipe/dancegrpo/run_dancegrpo.sh     # running
```
---

### 2. Reward Model

```
'dancegrpo_trainer.yaml'
    - reward_model.enable: True
    - reward_model.strategy: diffusion
    - reward_model.type: qwen/single/joint -> 1
    - reward_manager: dancegrpo -> 2
```
---

#### 1.reward_model.type
` 'Dance-grpo\recipe\dancegrpo\dancegrpo_fsdp_worker.py' `  
current reward model types:  
**a.qwen**  
    class QwenRewardModelWorker(RewardModelWorker)  
```
    # ---- running ----
    actor_rollout_ref.rollout.n # number of samples one prompt
    algorithm.adv_estimator=${adv_estimator} # 'grpo'
    reward_model.rollout.load_format=safetensors, # !!! safetensors 
    actor_rollout_ref.rollout.temperature, 
    actor_rollout_ref.rollout.top_p,  
    actor_rollout_ref.rollout.top_k,  
    actor_rollout_ref.rollout.val_kwargs.temperature,  
    actor_rollout_ref.rollout.val_kwargs.top_p,
    actor_rollout_ref.rollout.val_kwargs.top_k,
    actor_rollout_ref.rollout.tensor_model_parallel_size, # tp for rollout
    trainer.type, # diffusion
    # ---- implementation ----
    rollout.name # 'vllm'
    rollout.mode # 'sync'
    rollout.layered_summon # 'False'
    use_rm # 'True'
    # ---- others ----
    vllm_mode # 'spmd'
    world_size # Number of GPUs
    use_shm # 'True'
```

**b.single**  
    class DiffusionRewardModelWorker(RewardModelWorker)  
```

```

**c.joint**  
    class AestheticRewardModelWorker(RewardModelWorker),  
    class RAFTRewardModelWorker(RewardModelWorker),  
    class VideoclipRewardModelWorker(RewardModelWorker),  
    class VideophyRewardModelWorker(RewardModelWorker)  
    or  
    class MultiRewardModelWorker(RewardModelWorker) <- union four joint RewardModelWorkers
```

```

- If add a new reward model type, create a new class here and inherit from the class `RewardModelWorker`.
- Then,  
``` 
    from .dancegrpo_fsdp_worker import NewRewardModelWorker as RewardModelWorker  
    role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
```

#### 2.reward_manager
` 'verl\workers\reward_manager' @register('') `  
["BatchRewardManager", "DAPORewardManager", "NaiveRewardManager", "PrimeRewardManager", "AIGCRewardManager"]  
` class AIGCRewardManager -> @register('dancegrpo') `

- If add a new reward manager, create a new py file here and register it.