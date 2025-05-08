<div align="center">

Teletron
===========================
<h4>To pioneer training long-context multi-modal transformer models</h4>

[![version](https://img.shields.io/badge/release-0.1.0-green)](./setup.py)
[![license](https://img.shields.io/badge/license-Apache2.0-blue)](./LICENSE)

<div align="left">

## ⏱️Speed Benchmark 

- HunyuanVideo Training Speed 

Figure: Teletron 训练效率对比deepspeed，根据hunyuanvideo文章中说的渐进分辨率（256, 256, 65) -> (360, 640, 85) -> (540, 960, 105) -> (720, 1280, 129) ，GBS=8



## 📖Introduction

Teletron features flexible parallel strategy and fused cuda kernels to best facilitate **long-context**, **efficient** and **flexible** training of multi-modal transformer models.

* Long-Context
  * Teletron leverages mixed parallel strategy, activation checkpointing and fused cuda kernels at the same time to optimize GPU memory usage, so as to train [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) with up to 30s 720P video clips.
* Efficient
  * With async VAE and fused cuda kernels, Teletron facilitates faster training speed than general training optimization libraries like [DeepSpeed](https://github.com/deepspeedai/DeepSpeed).
* Flexible
  * Training with a variety of video sequence length and model size, Teletron support flexible adjustment of parallel strategy among data parallel, context parallel, and/or tensor parallel.

## ⚡️QuickStart

### Installation

```
git clone https://github.com/Tele-AI/Teletron.git
cd Teletron
# install fused kernels （optional）
bash teletron_op/install.sh
```

### Run pretraining

```
bash examples/vast/run.sh
```

Note that we include a snippet of the [Koala](https://github.com/KwaiVGI/Koala-36M) dataset in the repo. You may try with this tiny dataset or download full spec from the original repo.

### Run full-finetune

* Prepare pretrained weights
* Initiate training with t2i/t2v/i2v

### Inference

* Convert weights to huggingface format
* Inference as you like

## 🔥News

- 2025/5/16: Teletron First Release! Supports VAST/HunyuanVideo pretraining and full-tinetune!



## ✨Features

- [x] Ulysses Context Parallel

- [x] AdaLayerNorm fused kernel

- [x] RmsNorm fused kernel

- [ ] Asynchronous VAE 

- [ ] [Unified Sequence Parallel](https://arxiv.org/abs/2405.07719) 

  

## Acknowledgement

* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
* [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
* [Koala-36M](https://github.com/KwaiVGI/Koala-36M)
* [yunchang](https://github.com/feifeibear/long-context-attention)

## License

