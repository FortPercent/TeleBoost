# TeleBoost — Gradient Decoupled DPO 发布稿

> 三个版本：① 推特/标题级（30 字以内）② 公众号/博客中长版 ③ 技术博客/论文摘要长版
> 数据源全部来自 2026-04-30 32×H800 多机实测，详见 `MEMORY.md`。

---

## ① Headline / 一句话版（推特、内部 Slack、标语）

**EN**:
> TeleBoost open-sources Gradient Decoupled DPO — Wan 14B production-config
> DPO training that the standard implementation OOMs, now fits on 32×H800
> with 13 GB to spare.

**CN**:
> TeleBoost 开源 Gradient Decoupled DPO ——
> standard DPO 在 32×H800 上 OOM 跑不动的 Wan 14B 生产级训练配置
> （49 帧 480p），split DPO 不仅跑通，还留 37 GB 显存余量。

---

## ② 公众号 / 博客中长版（300-500 字）

### 标题候选
* TeleBoost 开源：让 Wan 14B 视频 DPO 真的能跑起来
* Gradient Decoupled DPO：用更小显存训更长视频
* 同样的硬件，多 8 倍序列：Gradient Decoupled DPO 实测

### 正文

DPO 是当下视频扩散模型对齐的主力方法，但在 14B 级别的 Wan I2V 模型上，
标准实现的显存峰值在 8×H800 80GB 上**直接 OOM**，连 production 默认的
49 帧 480p 都跑不起来。

我们今天开源 [TeleBoost](https://github.com/FortPercent/TeleBoost)：
基于 megatron-core 0.16.1 + DeepSpeed ZeRO-2 的 Wan 系列训练框架，
核心特性是 **Gradient Decoupled DPO**——对 chosen / rejected 两条 loss
分别 backward + 立即 reduce-scatter，让每条分支的全形梯度在下一次 backward
前就被释放成 1/N partition，从而把峰值显存压到一半以下。

实测数据（4 节点 ×8 H800 80GB，CP=8 DP=4，bf16 + ZeRO-2 + recompute=full）：

| 配置 | Standard DPO | Gradient Decoupled DPO |
|---|---|---|
| 49 帧 480p（~20k tokens） | ❌ OOM 79 GB | ✅ 42.91 GB |
| 81 帧 720p（~76k tokens） | ❌ | ✅ 48.72 GB |
| 81 帧 1080p（~171k tokens）| ❌ | ✅ 67.10 GB |
| 121 帧 1080p（~253k tokens）| ❌ | ❌ OOM |

**Decoupled DPO 在同样硬件上比 standard DPO 多吃 8.5× 的序列长度**。

数学上严格等价——我们对 36 层 14B Wan 在 32 GPU 上跑了一次 element-wise
对照（14.78 亿梯度元素），split 与单 backward 结果差异
`max|Δ|=2.4e-4`，远低于 bf16 ULP 阈值 1e-3。这是结构性等价
（reduce_scatter 是线性映射 + 链式法则），不是经验近似。

**仓库**：github.com/FortPercent/TeleBoost
**镜像**：基于 `nvcr.io/nvidia/pytorch:25.09-py3`，`docker build` 一键就绪
**文档**：[QUICKSTART.md](QUICKSTART.md) 5 分钟跑通 smoke test

---

## ③ 技术博客 / arXiv-style 摘要（800-1200 字）

### TeleBoost: Gradient Decoupled DPO for Memory-Efficient Video Diffusion Training

#### Background

Direct Preference Optimization (DPO) on video diffusion models like Wan
14B requires running both *chosen* and *rejected* branches through the
DiT before computing the preference loss
`L = −logσ(β·(loss_reject − loss_chosen))`. The naive implementation
materialises the loss as a single tensor and calls `loss.backward()`
once. With ZeRO-2 partitioned gradients, this leaves both branches'
full-shape gradients alive simultaneously during the entire reverse
pass — producing a memory peak that, on Wan 14B 40-layer at production
49-frame 480p config, exceeds the 80 GB capacity of an H800 even with
8-way context parallelism.

#### Method

**Gradient Decoupled DPO** restructures the backward pass:

```python
# Standard DPO  (full-grad of both branches alive at peak)
(coeff*(loss_chosen - loss_rejected)).backward()
optimizer.epilogue()

# Gradient Decoupled DPO  (per-branch backward + immediate reduce-scatter)
for t in [-coeff*loss_rejected, coeff*loss_chosen]:
    optimizer.backward(t)
    optimizer.overlapping_partition_gradients_reduce_epilogue()
```

The in-loop epilogue immediately reduce-scatters each branch's gradient
into the rank's 1/N partition, freeing the full-shape grad tensor before
the next backward starts. **Mathematically:**

```
my_slice(g_chosen) + my_slice(g_rejected)
  = my_slice(g_chosen + g_rejected)
  = my_slice(∇(loss_chosen + loss_rejected))            [chain rule]
```

— identical to single-backward of the summed loss, modulo bf16
rounding-order in the bucket accumulator.

#### Results

Hardware: 4 instances × 8×H800 80GB; CP=8 within node, DP=4 across; bf16
mixed-precision with fp32 Adam master; recompute-method=block,
recompute-num-layers=40; ZeRO-2 partition_grads=True (Adam state /32).

**Memory:**

| Config | Visual tokens | Standard | Decoupled | Δ |
|--------|---------------|----------|-----------|---|
| 49 f / 480p (production) | 20,280 | OOM 79 GB | **42.91 GB** | qualitative ✓ vs ✗ |
| 81 f / 720p | 75,600 | OOM | **48.72 GB** | qualitative |
| 81 f / 1080p | 171,360 | OOM | **67.10 GB** | qualitative |
| 121 f / 1080p | 252,960 | OOM | OOM (peaks ~79 GB) | both fail |

→ Standard DPO cannot fit production config; Decoupled DPO extends to
**~8.5× the sequence length** the standard implementation can handle on
the same hardware.

**Math equivalence:** verified element-wise on
`optimizer.averaged_gradients` (post reduce-scatter, pre `optimizer.step`)
across configs:

| Config | Elements compared | max|Δ| | Bit-identical |
|--------|-------------------|--------|---------------|
| 4-layer toy | 1.86 B | 1.37e-04 | 63.77 % |
| 36-layer 8 GPU fixture | 14.78 B | 2.44e-04 | 31.84 % |
| 36-layer 32 GPU prod-shape | 14.78 B | 2.44e-04 | 45.97 % |

All `max|Δ| ≪ 1e-3` (bf16 ULP for unit-scale gradients). The non-bit-identical
fraction comes from float-rounding-order differences in the bucket
reduce-scatter — irrelevant to numerical training stability.

#### Implementation

* Built on megatron-core 0.16.1 + DeepSpeed ZeRO-2 + flash-attn 3.
* Pinned to `deepspeed==0.17.5` — the last release with a multi-call-friendly
  epilogue (0.17.6+ refactored to require `DeepSpeedEngine`-driven
  `is_gradient_accumulation_boundary` toggling, which teletron, using the
  optimizer alone, doesn't provide).
* Drop-in: any user with a DPO dataset just subclasses
  `teletron.datasets.DPODatasetBase` and trains via `examples/teleai/train_dpo.sh`.
* `FakeDPODataset` ships for instant smoke-test without real video data.

#### Open source

Repository: github.com/FortPercent/TeleBoost
Branch: `dpo-wan-mc-0.16`
Quick-start: [QUICKSTART.md](QUICKSTART.md)
Docker base: `nvcr.io/nvidia/pytorch:25.09-py3`

#### Acknowledgments

Built on top of the Megatron-LM v0.16.1 release line
(github.com/NVIDIA/Megatron-LM/tree/core_v0.16.1), DeepSpeed
(deepspeedai/DeepSpeed), and flash-attention (Dao-AILab/flash-attention).
