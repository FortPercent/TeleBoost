# Dancegrpo

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
git clone https://github.com/hiahei/Dance_GRPO.git
cd Dance_GRPO

# install dependency
pip install -r requirements.txt

```

## 📊 Data Preprocess

```bash
    bash Dance-grpo\data_preprocess\preprocess_wan_rl_embeddings.sh
```

---

## 🏃 Usage

### 1. main script

```bash 
    bash Dance-grpo\recipe\dancegrpo\run_dancegrpo.sh
```

### 2. Reward Model Type

```
conifg file: 'Dance-grpo\recipe\dancegrpo\config\dancegrpo_trainer.yaml'
    reward_model.enable: True
    reward_model.strategy: diffusion
    reward_model.type: qwen/single/joint
```
```    
    a.qwen

    b.single

    c.joint
```
---

## 📝 TODO

* [x] 
* [√] 

## 📜 Citation