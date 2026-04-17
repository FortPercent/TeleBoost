# Feature Coverage Review — `main` (HEAD `13831488`)

**Date**: 2026-04-17
**Branch**: `main` on TeleBoost = `verl_0902` 框架 + wxe 业务代码 + smoke patches
**Test box**: 4 × H800 80GB

图例：✅ smoke 已验证 / 🟢 代码在且类型一致 / 🟡 代码在但未测 / 🔴 缺 / ⚠️ 有 caveat

---

## 1. Actor 模型规格

| 项 | 状态 | 说明 |
|---|---|---|
| Wan2.2-T2V-A14B (wan22) | ✅ | smoke #7 通过；代码 `dp_actor.py` / `diffusion_rollout.py` 有 `wan22_boundary` 专用分支 |
| Wan2.1-T2V-1.3B (wan21) | ✅ | smoke #8/#9/#11 通过 |
| Wan2.2-I2V-A14B (image-to-video) | 🟡 | 测试机有权重 (`/gfs/.../wxe/Wan-AI/Wan2.2-I2V-A14B`)，代码层未做 i2v 专属 smoke |
| Hunyuan T2V | 🟡 | `models/hunyuan/` 完整存在，`sample/sample_t2v_hunyuan.py` 入口在；非 dancegrpo 主流程 |
| Mochi | 🟡 | `sample/sample_t2v_mochi*.py`；同上 |
| 多模态（VLM 直接做 actor）| 🔴 | 不在范围 |

**结论**：dancegrpo 主流程对 Wan 系列（wan21/wan22, T2V）一等公民支持。i2v 没在 smoke 里跑过但应该可行（脚本直接换 model.path）。

---

## 2. Reward 类型

| Reward | 注册名 | 状态 | 说明 |
|---|---|---|---|
| HPSv2 (HPS_v2.1) | `hps` | ✅ | smoke #7/#8 通过；rewards=nan 是 smoke null context 占位副作用 |
| Aesthetic (CLIP+linear) | `aesthetic` | ✅ | smoke #11 worker init 通过 |
| RAFT (optical flow) | `raft` | ✅ | smoke #11 worker init 通过 |
| VideoCLIP-XL | `videoclip` | ✅ | smoke #11 worker init 通过 |
| Videophy (videocon_physics) | `videophy` | ✅ | smoke #11 worker init 通过（依赖 `decord` 已装） |
| Qwen-VL-7B (`type=qwen`) | — (内嵌 worker) | ✅ | smoke #9 真算出 reward (44→19.75，非 nan) |
| Qwen-VL-32B | — | 🟡 | 测试机有权重，但 smoke 没测；4 卡 + 1.3B actor 估计能装下 |
| Joint 4-reward (并行 weighted_sum) | `type=joint` | ✅ | smoke #11 通过；4 个 reward 都 emit metric |
| 自定义 reward function (Python callable) | `custom_reward_function` | 🟡 | yaml 有 `custom_reward_function.path` 字段，但当前 dancegrpo trainer 默认走 reward_model worker；若要插自定义 fn 需调整 |

**结论**：5 种插件 reward + 内嵌 Qwen + Joint 模式都覆盖。Qwen-VL-32B 没单独验证。

---

## 3. Rollout backend

| Backend | 目录/文件 | 状态 | 说明 |
|---|---|---|---|
| **diffusion_rollout** (Wan/Hunyuan/Mochi) | `verl/workers/rollout/diffusion_rollout.py` | ✅ | smoke 4 个都用这条 |
| flowgrpo_rollout | `verl/workers/rollout/flowgrpo_rollout.py` | 🟡 | 代码在；yaml `flow_grpo.enable=true` 默认开但 smoke 实际未触发 SDE 路径（sampling_steps=1 太小看不出） |
| mixgrpo_rollout | `verl/workers/rollout/mixgrpo_rollout.py` | 🟡 | 代码在；当前 yaml/脚本都没启用 |
| vllm_rollout (LLM 用) | `verl/workers/rollout/vllm_rollout/` | 🟡 | 不直接用于 diffusion actor；是 reward 模型（Qwen-VL）走的 backend |
| sglang_rollout | `verl/workers/rollout/sglang_rollout/` | 🟡 | 上游 verl 持续维护；本项目当前没用 |
| hf_rollout | `verl/workers/rollout/hf_rollout.py` | 🟡 | 调试用；smoke 没用 |
| naive_rollout | `verl/workers/rollout/naive/` | 🟡 | 上游 fallback；不用 |

**结论**：dancegrpo 主线只用 `diffusion_rollout` (actor) + `vllm_rollout` (Qwen reward)。flowgrpo/mixgrpo 代码在但 smoke 未真触发它们的核心路径。

---

## 4. Algorithm

| Algorithm | yaml/code trigger | 状态 |
|---|---|---|
| GRPO (group-relative PPO) | `algorithm.adv_estimator=grpo` | ✅ smoke 全部用这条 |
| GAE (vanilla PPO) | `adv_estimator=gae` | 🟢 上游 verl 一等公民 |
| GRPO-PassK | `adv_estimator=grpo_passk` | 🟢 上游 verl |
| RLOO | `adv_estimator=rloo` | 🟢 上游 verl |
| Reinforce++ | `adv_estimator=reinforce_plus_plus` | 🟢 上游 verl |
| OPO / REMAX | 同上 | 🟢 上游 verl |
| **GRPO-Guard** (wxe 算法) | `actor_rollout_ref.actor.grpo_guard.enable=true` | 🟡 yaml 字段在，default false；smoke 没启用，未实际验证 backward 对齐路径 |
| **flow-grpo / flow-grpo-fast** | `actor_rollout_ref.flow_grpo.enable=true` | 🟡 yaml 字段在，default true，但 sampling_steps=1 触发不到完整 SDE 路径 |

**结论**：标准 GRPO 链路 verified；wxe 加的 GRPO-Guard / flow-grpo 字段都在配置层，但**核心路径都没有 smoke 真实跑过**。

---

## 5. 部署

| 模式 | 状态 |
|---|---|
| 单机 1 GPU | 🔴 dancegrpo 全套（actor+ref+rollout+reward）单卡装不下 |
| **单机 4 GPU (本测试机)** | ✅ 1.3B actor 全部 smoke 通过 |
| 单机 8 GPU | 🟢 wxe 原版 3 个脚本默认；未在 4 卡机验证 |
| 多机 (NNODES>1) | 🟢 脚本支持 `NNODES=2 bash run_*.sh`；未验证（需要多节点环境） |

**结论**：4 卡能跑 1.3B；14B + reward 单步 4×80G 勉强能跑（smoke #7 verified）；8 卡和多机没在测试机验证。

---

## 6. 业务场景

| 场景 | 入口 | 状态 |
|---|---|---|
| **完整 RL 训练** | `recipe/dancegrpo/main_dancegrpo.py` | ✅ smoke 4 个都跑过，含 forward+backward+optim+ckpt |
| Validation (val_before_train) | `trainer.val_before_train` | 🟡 smoke 全部 `val_before_train=False` 跳过；未测 |
| Inference / Sample only | `sample/sample_t2v_*.py`, `generate.py` | 🟡 入口在；和 dancegrpo 训练流程独立 |
| Data preprocessing | `data_preprocess/preprocess_*.py` (10 个脚本) | 🟡 脚本在，smoke 用了 patched JSON 绕过；真实数据需要跑 `preprocess_wan_data.py` 生成含 `context_null_path` 的 JSON |
| Eval (单独 metric eval) | `verl/trainer/main_eval.py` | 🟡 上游 verl 提供；本项目没专属脚本 |
| Checkpoint resume | `trainer.resume_mode=auto` | ✅ smoke 验证过（曾误从前一次 ckpt resume，behavior 正确）|

**结论**：完整 RL 训练 + ckpt resume 通过；validation / sampling / 数据预处理流程**全部 untested**。

---

## 7. 运行依赖与脆弱点

### 7.1 必须的环境补丁（已在测试机生效，新机器要重做）

| # | 补丁 | 命令 |
|---|---|---|
| 1 | hpsv2 库 | `pip install hpsv2` (1.2.0) |
| 2 | hpsv2 BPE 词表 | `cp /gfs/.../wxe/HPSv2/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz /usr/local/lib/python3.10/dist-packages/hpsv2/src/open_clip/` |
| 3 | decord (videophy 需要) | `pip install decord` (0.6.0) |

副作用：1 把 protobuf 4.24 → 3.20（验证过 vllm/ray 仍正常）。

### 7.2 硬编码路径（启动脚本 hard-code，换机器要改）

| 资源 | 默认路径 | 修复方案 |
|---|---|---|
| Wan model | `/gfs/platform/public/infra/wxe/Wan-AI/Wan2.2-T2V-A14B` | 改用 `MODEL_PATH=` 环境变量化（脚本里 already 有 commented out 的 hint）|
| HPSv2 | `/gfs/platform/public/infra/models/HPS_v2.1_compressed.pt` | 同上 |
| 4 个 joint reward | `/gfs/platform/public/infra/models/{ViT-L-14.pt,raft-things.pth,VideoCLIP-XL.bin,videocon_physics}` | 同上 |
| Qwen-VL-7B | `/gfs/platform/public/infra/qrl760/Dance_GRPO/models/Qwen2.5-VL-7B-Instruct` | 同上 |
| Train data JSON | `/gfs/platform/public/infra/Dance-grpo/data/14B/rl_embeddings/processed_wan_prompt.json`（**测试机不存在！**）| 改为 `TRAIN_FILE=` 环境变量；新机器需重新预处理生成 |

### 7.3 Smoke 数据 vs 真实数据

| 项 | smoke | 真实 |
|---|---|---|
| `processed_wan_prompt.json` | wxe 旧版 (无 context_null_path) → patched 加占位 | 应跑 `data_preprocess/preprocess_wan_data.py` 生成完整新版 |
| `context_null.npy` | zeros (26, 4096) → 导致 reward = nan | 由预处理脚本用 T5 编码 negative prompt 生成 |
| Reward 数值 | nan (HPSv2/joint 都) 或不准 (Qwen-VL=44 是真值但因 batch 太小 advantage=0) | 应是合理 reward 分布 |

⚠️ 上线前**必须重新做数据预处理**，不能用 smoke 占位。

### 7.4 业务代码层 caveat（已 patch，commit `2167b04` + `b045935`）

| 位置 | 问题 | 已 patch |
|---|---|---|
| `dancegrpo_ray_trainer.py:741` | `pop(['caption','video_ids','video_frames'])` 强 assert，smoke 数据缺 video_ids/video_frames | defensive pop |
| `dancegrpo_ray_trainer.py:752 / 743` | 硬编码读 `gen_batch_output.batch['rewards']`，但 wxe reward worker 输出 `{model_name}_rewards` | union 后做 `rewards` alias |

⚠️ 这两个 patch 让真实数据训练**也更健壮**（即使数据齐全，reward worker 命名规则是 wxe 原生的，原代码也会因为命名不一致挂掉）。但 patch 是宽容性的，副作用应排查（比如 alias 的 'rewards' 是 hps 的 reward 还是 joint 的总 reward？验证一致性需要进一步代码 review）。

### 7.5 性能脆弱点

| 项 | 现状 |
|---|---|
| 14B FSDP 加载 | GFS 冷启每 shard ~28s ×6 = ~3 min；warm 缓存 ~7s/shard |
| Inductor 编译 | 1.3B 第一步 gen 700s（编译开销），warm 后 0.55s |
| ckpt save 大小 | 14B FSDP 4 rank 共 54GB；1.3B 几 GB |
| GFS write 速度 | ckpt save 14B 22s；1.3B 1.5s |

---

## 8. 整体支持度 dashboard

| 功能维度 | 已 verified | 代码在但未测 | 缺 |
|---|---|---|---|
| Actor 模型 | wan21 / wan22 | I2V, Hunyuan, Mochi | — |
| Reward | hps, aesthetic, raft, videoclip, videophy, qwen-7B, joint | qwen-32B, custom callable | — |
| Rollout | diffusion (actor), vllm (Qwen reward) | flowgrpo, mixgrpo, sglang, hf | — |
| Algorithm | GRPO | GRPO-Guard, flow-grpo, GAE, RLOO, etc. | — |
| 部署 | 单机 4 GPU | 单机 8 GPU, 多机 | 单机 1 GPU 无意义 |
| 业务 | 完整训练, ckpt resume | val_before_train, sampling, data preprocess | — |
| 数据 | smoke (zero null) | **真实数据全流程** | — |

**Verified vs Untested 比例**：Verified 覆盖了 dancegrpo 最核心的 RL 训练链路；untested 主要是变体（其他 actor / 其他 reward / 其他 rollout / 其他 algo） + 配套流程（validation / sample / preprocess）。

---

## 9. 接下来该做的"重新测试"建议清单

按优先级：

### P0：补充验证已有 patch 在真实数据下的行为
- [ ] 用 `data_preprocess/preprocess_wan_data.py` 生成真实带 `context_null_path` 的 JSON
- [ ] 重跑 smoke #7 / #11，验证 reward 不再 nan，advantage 是合理分布

### P1：smoke 覆盖目前 untested 的 dancegrpo 主路径变体
- [ ] **Qwen-VL-32B** smoke（用 1.3B actor + 32B reward, 看 4 卡显存是否够）
- [ ] **GRPO-Guard 启用** smoke（`grpo_guard.enable=true`，验证 wxe 自己开发的算法路径不挂）
- [ ] **flow-grpo SDE 启用** smoke（增大 sampling_steps 到 4-8 触发 SDE 路径）
- [ ] **val_before_train=True** smoke

### P2：业务流水线 e2e
- [ ] data preprocess 全流程（从 raw text → embedding → JSON）
- [ ] sample/inference 入口能 load 训练 ckpt 出视频

### P3：8 卡 / 多机部署
- 需要 8 卡测试机或多节点资源

---

## 10. 与"想支持的"对照

你列的目标：
1. 训练 actor: 14B / 1.3B / I2V / 多模态 — **14B+1.3B verified, I2V untested, 多模态不支持**
2. Reward: HPSv2 / Qwen-VL / Joint / 自定义 — **HPSv2/Qwen-VL/Joint verified, custom callable 未测**
3. Rollout: vllm / sglang / hf — **vllm verified (作 Qwen reward), sglang/hf untested**
4. Algorithm: GRPO / GRPO-Guard / flow-grpo — **GRPO verified, GRPO-Guard/flow-grpo untested**
5. 部署: 单机/多机 — **单机 4 GPU verified**
6. 业务: 完整训练 / 仅评估 / 数据预处理 — **训练 verified, 评估和预处理 untested**

**主要缺口**：GRPO-Guard 和 flow-grpo（wxe 自有算法）没真跑过；I2V 没跑过；自定义 reward callable 没跑过；validation/eval/preprocess 没跑过。
