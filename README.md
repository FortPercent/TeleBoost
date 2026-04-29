# DanceGRPO — Wan video diffusion + GRPO

基于 [verl](https://github.com/volcengine/verl) 框架的视频生成 RL 训练栈，actor 走 Wan2.1 / Wan2.2 文生视频模型，reward 支持 HPSv2 / Qwen-VL / 4-reward joint 三种方案，算法以 GRPO 为基线并扩展 GRPO-Guard / flow-grpo。

> 本仓库由 `verl_0902` 上游版本与 wxe 业务代码合并而来。完整决策与合并历史见 `docs/merge_records/`。

---

## 1. 当前能力速览

| 维度 | Verified | 仅代码在（未验证） |
|---|---|---|
| Actor | Wan2.2-T2V-A14B (wan22), Wan2.1-T2V-1.3B (wan21) | Wan2.2-I2V-A14B, Hunyuan, Mochi |
| Reward | HPSv2, Qwen-VL-7B, Joint(aesthetic+raft+videoclip+videophy) | Qwen-VL-32B, custom callable |
| Algorithm | GRPO | GRPO-Guard, flow-grpo SDE 路径, GAE/RLOO 等上游 |
| Rollout | diffusion (actor), vllm (Qwen reward) | sglang, hf, flowgrpo, mixgrpo |
| 部署 | 单机 4×H800 80G | 单机 8 GPU, 多机 |

完整 verified vs untested 矩阵：[`docs/review/feature_coverage.md`](docs/review/feature_coverage.md)

---

## 2. 快速开始（4×H800 80G 测试机）

### 2.1 环境一次性补丁

```bash
pip install hpsv2 decord
# HPSv2 pip 包遗漏 BPE 词表，从源码 cp 一份
cp $REPO_ROOT/recipe/dancegrpo/../HPSv2/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz \
   $(python3 -c "import hpsv2,os;print(os.path.dirname(hpsv2.__file__))")/src/open_clip/
```

### 2.2 跑 smoke

四个 4 卡缩水版 smoke 脚本（`n_gpus=4, steps=2, h=w=256, num_frames=9`），都在 `recipe/dancegrpo/`：

| 脚本 | 用途 |
|---|---|
| `run_dancegrpo_single_4gpu_smoke.sh` | 14B + HPSv2 |
| `run_dancegrpo_1p3B_4gpu_smoke.sh` | 1.3B + HPSv2 |
| `run_dancegrpo_1p3B_qwen_4gpu_smoke.sh` | 1.3B + Qwen-VL-7B |
| `run_dancegrpo_1p3B_joint_4gpu_smoke.sh` | 1.3B + 4-reward joint |

启动：

```bash
TRAIN_FILE=/path/to/processed_wan_prompt.json \
TEST_FILE=/path/to/processed_wan_prompt.json \
bash recipe/dancegrpo/run_dancegrpo_1p3B_4gpu_smoke.sh
```

测试机模型/数据真实路径见 [`docs/inventory/test_box_models.md`](docs/inventory/test_box_models.md)。

### 2.3 完整生产训练

参考 wxe 原版脚本（写死 8 卡 + 14B + 完整分辨率）：
- `recipe/dancegrpo/run_dancegrpo_single.sh`
- `recipe/dancegrpo/run_dancegrpo_qwen.sh`
- `recipe/dancegrpo/run_dancegrpo_joint.sh`

⚠️ 上线前需 `data_preprocess/preprocess_wan_data.py` 重新生成含 `context_null_path` 的训练 JSON（smoke 用 zeros 占位会导致 reward = nan）。

---

## 3. 仓库导航

```
recipe/dancegrpo/                       业务核心
├── main_dancegrpo.py                       入口
├── dancegrpo_ray_trainer.py                 RayPPOTrainer 子类
├── dp_actor.py                              DataParallelPPOActor 子类
├── dancegrpo_fsdp_worker.py                 7 个内嵌 Reward worker (Qwen/Diffusion/Aesthetic/RAFT/Videoclip/Videophy/Multi)
├── unified_reward_worker.py                 wxe 插件式 reward worker
├── reward_models/                           reward 插件注册表 (registry + 5 个插件 + composite + dynamic_joint)
├── config/dancegrpo_trainer.yaml            完整 hydra 配置 (含 grpo_guard, flow_grpo, joint.* 字段)
├── run_dancegrpo_*.sh                       生产 + smoke 启动脚本
└── ...

verl/                                   上游 verl 框架 (持续 rebase 自 verl 0.4.0.dev)
wan/                                    Wan 模型实现
qwen_reward/                            verl_0902 老的 Qwen reward 独立服务方案 (孤立, 不走主流程)
data_preprocess/                        数据预处理 pipeline
prompts/, models/, distill/, examples/  辅助资源
docs/                                   文档 (见下)
```

---

## 4. 文档

- [`docs/merge_records/merge_wxe_into_verl_0902.md`](docs/merge_records/merge_wxe_into_verl_0902.md) — 完整合并决策、4 次翻转历史、所有 patch 的来由、smoke 进度记录、4 卡资源估算
- [`docs/merge_records/wxe_reference/`](docs/merge_records/wxe_reference/) — 7 个 wxe 原版 Tier A 文件归档（cherry-pick 参考）
- [`docs/merge_records/wxe_patches/`](docs/merge_records/wxe_patches/) — wxe 独有 22 个 commits 的 patch 文件
- [`docs/inventory/test_box_models.md`](docs/inventory/test_box_models.md) — 测试机所有 actor / VAE / T5 / reward / data 路径与 size
- [`docs/review/feature_coverage.md`](docs/review/feature_coverage.md) — 功能 verified vs untested 完整矩阵 + 脆弱点分析

---

## 5. 分支结构（TeleBoost）

| 分支 | 内容 |
|---|---|
| `main` | **本分支**。verl_0902 + wxe 合并产物 + smoke 脚本 + 2 个代码 patch + 完整 docs |
| `hiahei_snapshot_20260417` | hiahei 在测试机 `/gfs/.../wxe/Dance-grpo` 的工作快照（含 21 个未提交改动） |

---

## 6. 已知 wxe 业务代码 patch（在 main 上）

| Commit | 文件 | 修复 |
|---|---|---|
| `2167b04` | `recipe/dancegrpo/dancegrpo_ray_trainer.py:741` | reward 后清理 batch 时改为 defensive pop（原代码硬要 `caption/video_ids/video_frames` 三个 key 都在）|
| `b045935` | 同上文件 :752 | reward worker 输出 `{model_name}_rewards`，下游 alias 为 `rewards`（原代码硬编码读 `rewards` 会 KeyError）|

两个 patch 让真实数据训练也更健壮（不只是 smoke 数据需要）。

---

## 附录：FLUX 训练经验（来自原 hiahei README）

> 以下内容来自项目早期 FLUX 训练经验（已不再是当前主线，actor 已切到 Wan）。

1. We set the inference batch size to 1 because we observed differences in probability outputs when it exceeds the training batch size.
2. A stronger SFT stage can suppress exploration during the GRPO phase.
3. For extreme cases (same prompt + same initial noise + reward can't distinguish), try varying initial noise within a prompt.
4. Extended training (larger `max_train_steps`) may not improve visualization quality due to reward model limits (HPS-v2.1 not optimized for FLUX). EMA support is planned.
