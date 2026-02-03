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
```bash 
    bash recipe\dancegrpo\run_dancegrpo.sh
```

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

**b.single**  
    class DiffusionRewardModelWorker(RewardModelWorker)

**c.joint**  
    class AestheticRewardModelWorker(RewardModelWorker),  
    class RAFTRewardModelWorker(RewardModelWorker),  
    class VideoclipRewardModelWorker(RewardModelWorker),  
    class VideophyRewardModelWorker(RewardModelWorker)  
    or  
    class MultiRewardModelWorker(RewardModelWorker) <- union four joint RewardModelWorkers

- If add a new reward model type, create a new class here and inherit from the class `RewardModelWorker`.

#### 2.reward_manager
` 'verl\workers\reward_manager' @register('') `  
["BatchRewardManager", "DAPORewardManager", "NaiveRewardManager", "PrimeRewardManager", "AIGCRewardManager"]  
` class AIGCRewardManager -> @register('dancegrpo') `

- If add a new reward manager, create a new py file here and register it.

---

## 📝 TODO

* [x] 
* [√] 

## 📜 Citation