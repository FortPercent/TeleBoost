<div align="center">

TeleTron
===========================
<h4>To pioneer training long-context multi-modal transformer models</h4>

[![version](https://img.shields.io/badge/release-0.1.0-green)](./setup.py)
[![license](https://img.shields.io/badge/license-Apache2.0-blue)](./LICENSE)

<div align="left">

## ⏱️Speed Benchmark 

- HunyuanVideo Training Speed 

Figure: TeleTron 训练效率对比deepspeed，根据hunyuanvideo文章中说的渐进分辨率（256, 256, 65) -> (360, 640, 85) -> (540, 960, 105) -> (720, 1280, 129) ，GBS=8



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

To save efforts on environment setup, it is recommended using [nvcr](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)'s 24.10-py3 container image. 

```
# pull docker image
docker pull nvcr.io/nvidia/pytorch:24.10-py3

# start docker container
sudo docker run --gpus all -itd --shm-size 512G --name litian  nvcr.io/nvidia/pytorch:24.10-py3 /bin/bash

# enter the container
sudo docker exec -it litian /bin/bash
```

In the docker container, follow the script below to setup TeleTron.

```
# get TeleTron
git clone git@github.com:AI-Infra-Team/TeleTron.git --recurse-submodule

# install requirements
pip install -r requirements.txt

# (optional) install TeleTron fused kernels 
cd teletron_op && bash install.sh
```

### Sanity Check

The script below will run a tiny version of HunyuanVideo with fake data. It serves as a sanity check for that the environment is correctly set up.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=12345 bash examples/vast/run_unified_sanity_check.sh 1 1
```

### Training

* single node training

```
bash examples/vast/run_unified.sh 2 2 9
```

Note that TeleTron the trailing numbers designates TP size, CP size, and the number of frames.  You may also alter training video resolution with `--video-resolution {width} {height}` . 

* Multi-node training

Run the script below on 4 * 8 H800 cluster and 129-frame 720P training will be initiated. Note that for full finetuning you still need to download and convert HunyuanVideo pretrained weights.

```
bash examples/vast/run_unified.sh 1 4 129
```

## 🔥News

- 2025/5/16: TeleTron First Release! Supports HunyuanVideo finetuning and inference.

## ✨Features

- [x] Ulysses Context Parallel

- [x] AdaLayerNorm fused kernel

- [x] RmsNorm fused kernel

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

