<div align="center">

TeleTron
===========================
<h4>To pioneer training long-context multi-modal transformer models</h4>

[![version](https://img.shields.io/badge/release-0.1.0-green)](./setup.py)
[![license](https://img.shields.io/badge/license-Apache2.0-blue)](./LICENSE)

<div align="left">


## 🔥News

- 2025/5/16: TeleTron First Release! Supports HunyuanVideo finetuning and inference.


## 📖Introduction

TeleTron features flexible parallel strategy and fused cuda kernels to best facilitate **long-context**, **efficient** and **flexible** training of multi-modal transformer models.

* Long-Context
  * TeleTron leverages mixed parallel strategy, activation checkpointing and fused cuda kernels at the same time to optimize GPU memory usage, so as to train [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) with up to 30s 720P video clips.
* Efficient
  * With fused cuda kernels, TeleTron facilitates faster training than general training optimization libraries like [DeepSpeed](https://github.com/deepspeedai/DeepSpeed).
* Flexible
  * Training with a variety of video sequence length and model size, TeleTron support flexible adjustment of parallel strategy among data parallel, context parallel, and/or tensor parallel.

## ⚡️QuickStart

### Installation

Docker image tag: `harbor.telecom-ai.com.cn/teleai-t2v/ncvr-torch2.5-cuda12.4:v0.2`

In the docker container, follow the script below to setup TeleTron.

```
# get TeleTron
git clone ssh://${your_user_name}@code.srdcloud.cn:29418/P24HQASYF0004/AI-Infra/Teletron

# install requirements
pip install -r requirements.txt

# install TeleTron fused kernels 
cd teletron_op && bash install.sh
```

### Sanity Check

The script below will run a tiny version of HunyuanVideo with fake data. It serves as a sanity check for that the environment is correctly set up.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=12345 bash examples/vast/run_unified_sanity_check.sh 1 1
```

### Training

* single node training

The script below starts one-node training of HunyuanVideo/VAST. The default training setting is i2v, 720P 49 frames. 

```
bash examples/vast/run_unified.sh 2 2
```

Note that the trailing numbers designates TP size and CP size respectively. You may see the full set of training options in `examples/vast/pretrain_hunyuanvideo.py`. Note that for full finetuning you still need to download and convert HunyuanVideo pretrained weights.

* Multi-node training

```
bash examples/vast/run_unified.sh 1 4 
```

If using k8s, just start a 4-node pod and run the script above. Otherwise, you need to set the env variables below before starting training.
```
$MASTER_ADDR
$RANK
$WORLD_SIZE
```


## ✨Features

- [x] Ulysses Context Parallel

- [x] AdaLayerNorm fused kernel

- [x] RmsNorm fused kernel

- [x] Support VAST dataloader (thanks to yxy)

- [ ] Asynchronous VAE 

- [ ] [Unified Sequence Parallel](https://arxiv.org/abs/2405.07719) 

## Acknowledgement

* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
* [Diffusers](https://github.com/huggingface/diffusers)
* [yunchang](https://github.com/feifeibear/long-context-attention)
* [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
* [Koala-36M](https://github.com/KwaiVGI/Koala-36M)

## License

[Apache 2.0 License](https://github.com/AI-Infra-Team/TeleTron/blob/main/LICENSE)

