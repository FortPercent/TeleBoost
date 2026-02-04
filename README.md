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
#### reward_model.enable: True
##### 1.reward_model.type
` 'Dance-grpo\recipe\dancegrpo\dancegrpo_fsdp_worker.py' `  
current reward model types:  
**a.qwen**  
    class QwenRewardModelWorker(RewardModelWorker)  
```
    # ---- running ----
    rewardmodel.model.path='path/to/Qwen2.5-VL-7B-Instruct'
    actor_rollout_ref.rollout.n # number of samples one prompt
    algorithm.adv_estimator=${adv_estimator} # 'grpo'
    reward_model.rollout.load_format=safetensors, 
        # !!! for qwen 'load_format' must be 'safetensors' 
    actor_rollout_ref.rollout.temperature, 
    actor_rollout_ref.rollout.top_p,  
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
    # running
    reward_model.model.path='path/to/HPSv2/HPS_v2_compressed.pt'
```    

**c.joint**  
    class AestheticRewardModelWorker(RewardModelWorker),  
    class RAFTRewardModelWorker(RewardModelWorker),  
    class VideoclipRewardModelWorker(RewardModelWorker),  
    class VideophyRewardModelWorker(RewardModelWorker)  
    or  
    class MultiRewardModelWorker(RewardModelWorker) <- union four joint RewardModelWorkers
```
    # implementation
    reward_model:  
        type: joint  
        aesthetic:  
            clip_model_path: /path/to/ViT-L-14.pt  
            aes_model path: /path/to/sa_0_4_vit_l_14_linear.pth  
        raft:  
            model_path: /path/to/raft-things.pth  
        videoclip:  
            model path: /path/to/VideoCLIP-XL.bin  
        videophy:  
            model_path: /gemini/space/wyb/model/arena_model/videocon_physics  
```

- If add a new reward model type, create a new class here and inherit from the class `RewardModelWorker`.
- Then,  
``` 
    from .dancegrpo_fsdp_worker import NewRewardModelWorker as RewardModelWorker  
    role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
```

##### 2.reward_manager
` 'verl\workers\reward_manager' @register('') `  
["BatchRewardManager", "DAPORewardManager", "NaiveRewardManager", "PrimeRewardManager", "AIGCRewardManager"]  
` class AIGCRewardManager -> @register('dancegrpo') `

- If add a new reward manager, create a new py file here and register it.

#### reward_model.enable: False
```
use_rm: False
    1) To integrate a custom reward function, simply define the script path in 'verl/trainer/config/ppo_trainer.yaml'. The system will automatically register this function and use it as 'compute_score'.
        custom_reward_function:
            path: path/to/your_script.py   # Path to your .py file
            name: your_function_name       # Function name to use as compute_score
```
```
    2) If not define, directly use 'default_compute_score' in 'verl\trainer\ppo\reward.py' for various datasets.
```