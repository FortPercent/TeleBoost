# Test Box Model / Data Inventory

**Box**: `ssh -p 30022 'L5bA1R9X@root@ssh-434.default@222.223.106.147'`
**Hardware**: 4 × NVIDIA H800 80GB (320G total), Python 3.10.12, torch 2.6.0+cu124, vllm 0.8.4, ray 2.43.0
**Last verified**: 2026-04-17

---

## 1. Actor (Text-to-Video 主模型)

| Model | Path | Size | Version | Notes |
|---|---|---|---|---|
| **Wan2.2-T2V-A14B** | `/gfs/platform/public/infra/wxe/Wan-AI/Wan2.2-T2V-A14B` | 118G | wan22 | wxe 部署，smoke #7 用这个 |
| Wan2.2-T2V-A14B (dup) | `/gfs/platform/public/infra/qrl760/Dance_GRPO/models/Wan2.2-T2V-A14B` | 171G | wan22 | qrl760 的副本（更大可能含 optimizer state） |
| Wan2.2-I2V-A14B | `/gfs/platform/public/infra/wxe/Wan-AI/Wan2.2-I2V-A14B` | 118G | wan22 | image-to-video 变体，目前没用 |
| **Wan2.1-T2V-1.3B** | `/tmp/Wan2.1-T2V-1.3B` | 17G | wan21 | HF snapshot_download 下载的，smoke #8 用 |

## 2. VAE

同 Wan 主模型目录下：
- 14B: `/gfs/platform/public/infra/wxe/Wan-AI/Wan2.2-T2V-A14B/Wan2.1_VAE.pth` (485M)
- 1.3B: `/tmp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`（同上 VAE 规格）

## 3. T5 Text Encoder（UMT5-XXL）

| 来源 | 路径 |
|---|---|
| 1.3B 自带 | `/tmp/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth`（bf16, 含 google tokenizer） |
| 14B 目录下 | 同 actor 路径（未单独核实） |

## 4. Reward Models

### 4.1 HPSv2（smoke single 用）

| 资源 | 路径 | Size |
|---|---|---|
| 权重 `HPS_v2.1_compressed.pt` | `/gfs/platform/public/infra/models/HPS_v2.1_compressed.pt` | 1.9G |
| HPSv2 源码 + BPE 词表 | `/gfs/platform/public/infra/wxe/HPSv2/` | 22M |
| `bpe_simple_vocab_16e6.txt.gz` | `/gfs/platform/public/infra/wxe/HPSv2/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz` | 1.4M |
| pip 包 | `pip install hpsv2` (1.2.0) → `/usr/local/lib/python3.10/dist-packages/hpsv2/` | — |

> ⚠️ pip 包缺 BPE 词表，需手动 `cp` 源码那份到 `<site-packages>/hpsv2/src/open_clip/`

### 4.2 Qwen-VL（smoke qwen 用）

| Model | Path | Size |
|---|---|---|
| Qwen2.5-VL-32B-Instruct | `/gfs/platform/public/infra/Qwen2.5-VL-32B-Instruct` | 64G |
| Qwen2.5-VL-32B-Instrut (typo dup) | `/gfs/platform/public/infra/Qwen2.5-VL-32B-Instrut` | — |
| **Qwen2.5-VL-7B-Instruct** | `/gfs/platform/public/infra/qrl760/Dance_GRPO/models/Qwen2.5-VL-7B-Instruct` | 16G |
| Qwen3-VL-8B-Instruct | `/gfs/platform/public/infra/Qwen3-VL-8B-Instruct` | 17G |
| Qwen3-VL-30B-A3B-Instruct (MoE) | `/gfs/platform/public/infra/Qwen3-VL-30B-A3B-Instruct` | 58G |

> `/gfs/platform/public/infra/Qwen2.5-VL-7B-Instruct`（不带 qrl760 前缀）**不存在**——别被文件名骗。

### 4.3 Joint 4-reward（smoke joint 用）

全部在 `/gfs/platform/public/infra/models/`:

| Reward | 文件 | Size |
|---|---|---|
| Aesthetic - CLIP backbone | `ViT-L-14.pt` | 890M |
| Aesthetic - linear head | `sa_0_4_vit_l_14_linear.pth` | 32K |
| RAFT (optical flow) | `raft-things.pth` | 21M |
| VideoCLIP-XL | `VideoCLIP-XL.bin` | 1.6G |
| Videophy (videocon_physics) | `videocon_physics/` | 14G |

## 5. Training Data

| 资源 | 路径 | Size | Notes |
|---|---|---|---|
| **Prompt JSON (wxe 制作)** | `/gfs/space/chatrl/users/wxe/fastvideo/data/processed_wan_prompt.json` | 384K | 2000 条 `{caption, context_path}`; **缺 `context_null_path` 字段**，新版 dataset 需要 |
| 每条 prompt 的 T5 embedding | `/gfs/space/chatrl/users/wxe/fastvideo/data/context_*.npy` | ~638M 总 | shape (26, 4096) float32 UMT5-XXL |
| Smoke 用 patched JSON | `/tmp/processed_wan_prompt_smoke.json` | 442K | 给每条补了 `context_null_path` 指向 zeros npy |
| Smoke 用 zero null npy | `/tmp/context_null_smoke.npy` | 416K | shape (26, 4096) float32 全 0（占位，reward 会出 nan） |
| 脚本中写的路径（**不存在**）| `/gfs/platform/public/infra/Dance-grpo/data/14B/rl_embeddings/processed_wan_prompt.json` | — | 原 single/qwen/joint.sh 硬编码的 GFS 路径，在测试机无此结构 |

> 真实训练时需用 `data_preprocess/preprocess_wan_data.py` 跑一遍，生成带真 `context_null_path`（neg prompt 的 UMT5 embedding）的 JSON，不能用 smoke 的 zeros 版。

## 6. Smoke 启动脚本 → 用了哪些资源

| 脚本 | actor | reward | 数据 | 状态 |
|---|---|---|---|---|
| `run_dancegrpo_single_4gpu_smoke.sh` | Wan2.2-14B | HPSv2 | smoke JSON | ✅ smoke #7 通过 |
| `run_dancegrpo_1p3B_4gpu_smoke.sh` | Wan2.1-1.3B | HPSv2 | smoke JSON | 🟡 smoke #8 跑中 |
| `run_dancegrpo_1p3B_qwen_4gpu_smoke.sh` | Wan2.1-1.3B | Qwen2.5-VL-7B | smoke JSON | ⏳ 待跑 |
| `run_dancegrpo_1p3B_joint_4gpu_smoke.sh` | Wan2.1-1.3B | 4-reward joint | smoke JSON | ⏳ 待跑 |
| `run_dancegrpo_single.sh`（原版）| Wan2.2-14B | HPSv2 | 脚本写的不存在路径 | ❌ 没跑（硬编码 n_gpus=8） |
| `run_dancegrpo_qwen.sh`（原版）| Wan2.2-14B | Qwen-VL-**32B** | 同上 | ❌ 没跑 |
| `run_dancegrpo_joint.sh`（原版）| Wan2.2-14B | 4-reward joint | 同上 | ❌ 没跑 |

## 7. 环境修复（一次性）

在测试机执行过（会污染系统 Python，后续在该机器上持续有效）：

```bash
pip install hpsv2                    # 1.2.0, 附带 timm/webdataset/clint/braceexpand；protobuf 4.24→3.20（无副作用）
cp /gfs/platform/public/infra/wxe/HPSv2/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz \
   /usr/local/lib/python3.10/dist-packages/hpsv2/src/open_clip/
```

---

**维护备注**：这份 inventory 是 wiki-page 的一个实例（参考 Karpathy LLM Wiki 模式）。新增模型或路径变动时追加记录；做了 smoke 后标注"哪条路径实际被跑通"。
