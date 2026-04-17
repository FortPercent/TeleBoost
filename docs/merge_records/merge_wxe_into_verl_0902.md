# Merge Record: `dance-grpo/wxe` → `verl_0902` (→ branch `verl_0902_wxe`)

- **日期**：2026-04-17
- **操作者**：@samanthazzz929
- **源分支**：`dance-grpo/wxe`（仓库 `/Users/wuxuaner/Desktop/teleai/Dance-grpo`，HEAD `ed3caabd`）
- **目标分支**：`origin/verl_0902`（当前仓库，HEAD `3109719f`）
- **产物分支**：`verl_0902_wxe`（merge commit `c4f237d0`）

---

## 1. 背景与前置判断

### 1.1 两分支关系（非 fork，独立历史）

| 维度 | `origin/verl_0902` | `dance-grpo/wxe` |
|---|---|---|
| 所在仓库 | `AI-Infra/Dancegrpo`（无连字符） | `AI-Infra/Dance-grpo`（带连字符） |
| HEAD | `3109719f` (Xuaner, 2026-02-03) | `ed3caabd` (hiahei, 2026-02-11) |
| commit 数 | 937 | 80 |
| 血缘 | 基于 verl 上游 0.4.0.dev + 持续 rebase + wan/reward feature | 本地独立 `git init`，一次性导入 verl 0.4.0.dev 代码 |

- `git merge-base origin/verl_0902 dance-grpo/wxe` → **空**（无共同祖先）
- 两分支共同 commit-tree SHA = 0，共同 blob SHA = 16（全是上游 verl 原封不动的文件：`Notice.txt`、docs、`bpe_simple_vocab_16e6.txt.gz` 词表等，核心代码 0 共享）
- 路径级：771 个共同路径中，仅 24 个（3%）内容完全一致，747 个（97%）被各自独立修改

### 1.2 开发方向差异

- **verl_0902**：上游 verl 持续 rebase + wan22 集成 + 多 reward model（hpsv2）+ torch compile + OOM/offload 修复；reward 改造走 `qwen_reward/` 独立服务化方向
- **wxe**：GRPO-Guard / flow-grpo 算法对齐（反复修 log_prob）+ reward 插件化（`recipe/dancegrpo/reward_models/`）+ `unified_reward_worker.py` 统一 worker

两边 reward 重构**方向不同但路径不冲突**（`qwen_reward/` vs `recipe/dancegrpo/reward_models/`），本可共存；冲突集中在两边都重写过的 `recipe/dancegrpo/*.py` 核心训练代码。

---

## 2. 执行过程

```bash
git checkout -b merge_wxe_tmp origin/verl_0902
git merge dance-grpo/wxe --allow-unrelated-histories --no-commit --no-ff
```

结果：**744 个 add/add 冲突**（add/add 是因为两分支无共同祖先，同路径文件在 git 眼里都是"新增"）。

### 2.1 冲突按目录分布

```
 282 verl/           （上游 verl 源码，两边各自改过的部分）
 105 tests/
 104 examples/
  66 recipe/
  50 docs/
  43 models/
  26 wan/
  11 utils/
  10 docker/
   8 data_preprocess/
   6 qwen_reward/     （verl_0902 独有目录，冲突是因为 wxe 空目录/不同）
   其他杂项 ×30+
```

### 2.2 冲突结构特征

检查 Tier A 7 个文件的冲突块数：6 个文件是**单一大冲突块**（`<<<<<<<` 在开头 `>>>>>>>` 在末尾），1 个文件（`dancegrpo_ray_trainer.py`）有 14 个块但含 443/161/159 行的超大块。

**含义**：两边的同路径文件内容完全不同，git auto-merge 无法做行级对齐，本质是**整文件二选一**，不存在"混合两边的行"的有效选项。

---

## 3. 解决策略：分层处理

### Tier A — 业务核心（7 个，需判断）

| 文件 | 冲突块数 | 处理 |
|---|---|---|
| `recipe/dancegrpo/config/dancegrpo_trainer.yaml` | 1 | `ours` |
| `recipe/dancegrpo/dancegrpo_fsdp_worker.py` | 1 | `ours` |
| `recipe/dancegrpo/dancegrpo_ray_trainer.py` | 14（含 443/161 行巨块）| `ours` |
| `recipe/dancegrpo/dp_actor.py` | 1 | `ours` |
| `recipe/dancegrpo/main_dancegrpo.py` | 1 | `ours` |
| `verl/trainer/ppo/ray_trainer.py` | 1 | `ours` |
| `verl/workers/rollout/diffusion_rollout.py` | 1 | `ours` |

### Tier B — 上游/附属（737 个，批量处理）

`wan/`、`verl/`（除 Tier A）、`tests/`、`examples/`、`docs/`、`models/`、其他 `recipe/*`（dapo / prime / r1 / spin 等 wxe 不涉及的上游 recipe）、顶层配置（`.gitignore`、`requirements*.txt`、`pyproject.toml`、`README.md`、`LICENSE`）、`docker/`、`qwen_reward/`、`utils/`、`scripts/`、`sample/`、`distill/`、`config_sd/` 等。

执行命令：

```bash
cat /tmp/tier_b.txt | xargs -I {} git checkout --ours -- "{}"
cat /tmp/tier_b.txt | xargs git add
cat /tmp/tier_a.txt | xargs -I {} git checkout --ours -- "{}"
cat /tmp/tier_a.txt | xargs git add
```

---

## 4. 为什么 Tier A 也选 `ours`（verl_0902 版）

1. **不可行的选项：逐块混合**
   - 6/7 文件是单一大冲突块，根本没有可混合的粒度。
   - 唯一多块的 `dancegrpo_ray_trainer.py` 含 443 行训练循环巨块，单块内部依然是二选一。

2. **选 `ours`（verl_0902）胜过选 `theirs`（wxe）**
   - verl_0902 基于更新的 verl 上游（2025 年持续 rebase，含 OOM / offload / torch compile 修复）
   - verl_0902 已集成 wan22、multi reward model、hpsv2、torch compile、训练指标（clip fraction、grad norm、advantage、time）
   - wxe 基于较早的 verl 快照，在一次性导入后本地演化，算法修复（log_prob / GRPO_guard）虽有价值但与 verl_0902 的新代码结构不兼容，字面覆盖会丢失大量上游改进

3. **`theirs` 策略会丢失的价值（verl_0902 独有）**
   - 164 个 verl_0902 独有文件（wan22、新 verl 模块、sglang/megatron 新工具）
   - `qwen_reward/` 整套 Qwen VLM reward 服务实现
   - 所有 torch compile 与 offload/OOM 优化

4. **`ours` 策略不会丢的 wxe 价值**
   - wxe 的**新增文件**（非冲突，auto-merged）已全部合入 → 见 §5
   - wxe 的**算法修复**（log_prob 对齐、GRPO_guard backward）**未字面合入**，但保留为后续 cherry-pick 待办 → 见 §6

---

## 5. 实际合入的 wxe 新增（21 个文件，非冲突自动合并）

```
A  data_preprocess/preprocess_wan_data.py
A  data_preprocess/preprocess_wan_rl_embeddings.sh
A  prompts/hard_50.txt
A  prompts/istock_2000.txt
A  prompts/istock_5w.txt              ← 约 5 万行训练 prompt
A  prompts/mini_test.txt
A  recipe/dancegrpo/reward_models/__init__.py
A  recipe/dancegrpo/reward_models/aesthetic.py
A  recipe/dancegrpo/reward_models/base.py        ← 基类 457 行
A  recipe/dancegrpo/reward_models/composite.py   ← 384 行
A  recipe/dancegrpo/reward_models/dynamic_joint.py
A  recipe/dancegrpo/reward_models/hps.py
A  recipe/dancegrpo/reward_models/raft.py
A  recipe/dancegrpo/reward_models/registry.py    ← 102 行注册机制
A  recipe/dancegrpo/reward_models/videoclip.py
A  recipe/dancegrpo/reward_models/videophy.py
A  recipe/dancegrpo/run_dancegrpo_joint.sh
A  recipe/dancegrpo/run_dancegrpo_qwen.sh        ← wxe 里 rename 自 run_dancegrpo.sh
A  recipe/dancegrpo/run_dancegrpo_single.sh
A  recipe/dancegrpo/unified_reward_worker.py     ← 637 行统一 worker
A  utFile                                        ← 0 字节空文件（wxe 残留）
```

注意：`run_dancegrpo_qwen.sh` 在 wxe 里是由 `run_dancegrpo.sh` 重命名而来；verl_0902 的 `run_dancegrpo.sh` 保留，两脚本目前共存，内容不同。

---

## 6. Tier A 的 wxe 改动（经两次翻转后：**仍为 verl_0902 版**）

> **历史（保留作记录）**：
> - 首次 merge（`c4f237d0`, `ours`）：7 个 Tier A 文件取 verl_0902 版
> - 第二阶段（`f9c44a4f`）：应请求改为 wxe 版整文件覆盖（二次合并吸收 wxe 算法修复）
> - 校验后回退（`b1553b86` + `ed554da8`）：按 "以 verl_0902 为主" 原则，7 个 Tier A **全部回到 verl_0902 版**
>
> **最终实质效果**：分支 = verl_0902 + wxe 的 21 个**纯新增**文件（reward_models/, unified_reward_worker.py, prompts/, 新启动脚本）。wxe 的 **Tier A 算法修复（log_prob 对齐 / GRPO_guard backward / diffusion rollout fix）未合入**。
>
> 被 f9c44a4f 短暂覆盖过的 verl_0902 文件可通过 `git show c4f237d0:<path>` 确认未丢失；
> wxe 版同一文件可通过 `git show dance-grpo/wxe:<path>` 取出，作为后续手动 cherry-pick 参考。

### wxe Tier A 文件保留的价值（未合入，待手动 cherry-pick）

以下改动存在于 wxe 的 Tier A 文件中，被 `ours` 策略丢弃，需作为独立任务手动 patch 到 verl_0902 版本的代码上：

| 来源文件（wxe 中） | 需要移植的改动 | 对应 commit 线索 |
|---|---|---|
| `recipe/dancegrpo/dp_actor.py` | log_prob 计算维度对齐修复 | `2084c82 解决了log_prob没有对齐的问题`（+ 连续 8 个 `修改了额计算log_prob的逻辑`） |
| `recipe/dancegrpo/dancegrpo_ray_trainer.py` | GRPO_guard backward 对齐；flow-grpo / flow-grpo-fast 集成点 | `dba4a41 GRPO_guard的backward对齐了`、`34202ae refactor and add gaurd-grpo / flow-grpo / flow-grpo-fast` |
| `recipe/dancegrpo/dancegrpo_fsdp_worker.py` | 调用 `unified_reward_worker` 的接入点 | `216cab6`、`3cb9107 refactor reward function` |
| `recipe/dancegrpo/main_dancegrpo.py` | 新启动脚本的入口适配 | `e8ad31c refactor reward` |
| `recipe/dancegrpo/config/dancegrpo_trainer.yaml` | reward_models 插件化配置字段 | `dc8770a 增加了config文件的注释` |
| `verl/trainer/ppo/ray_trainer.py` | wxe 在 PPO trainer 上的 GRPO-Guard 相关钩子 | `578727e 挪动了dancegrpo_ray_trainer的位置` 引起的连带改动 |
| `verl/workers/rollout/diffusion_rollout.py` | diffusion rollout 的 log_prob / 形状修复 | `9dee20b 遇到了tensor形状问题 from dp_actor` |

### 获取 wxe 原版文件的方法

```bash
# 查看 wxe 版单个文件内容
git show dance-grpo/wxe:recipe/dancegrpo/dp_actor.py

# 生成 wxe 版与 verl_0902 版的 diff（供 patch 参考）
git diff origin/verl_0902:recipe/dancegrpo/dp_actor.py dance-grpo/wxe:recipe/dancegrpo/dp_actor.py

# 把 wxe 版单文件 checkout 到工作目录（慎用：会覆盖现有版）
git checkout dance-grpo/wxe -- recipe/dancegrpo/dp_actor.py
```

### 第二阶段覆盖的风险与后续

**丢失的 verl_0902 改动**（在这 7 个文件内）：
- `recipe/dancegrpo/dancegrpo_fsdp_worker.py`：verl_0902 里的 `qwen_reward` 服务调用入口、torch compile 集成点、offload/OOM 修复 → **覆盖后丢失**
- `recipe/dancegrpo/dp_actor.py`：verl_0902 的 offload / compile 相关改动
- `verl/trainer/ppo/ray_trainer.py`：上游 verl rebase 的 PPO trainer 更新
- `verl/workers/rollout/diffusion_rollout.py`：上游 verl rebase 的 diffusion rollout 修复

**需人工验证或二次调整**：
1. `qwen_reward/` 目录仍然存在，但 `dancegrpo_fsdp_worker.py` 已是 wxe 版（不知道 qwen_reward），若要同时用 verl_0902 的 qwen reward 服务和 wxe 的 reward_models 插件，需在 wxe 版 worker 里**重新加入** `qwen_reward` 调用点
2. 跑通性验证：wxe 的 `dancegrpo_fsdp_worker.py` 可能引用了 verl_0902 没有的某些上游符号，或引用了 verl_0902 重构后改名的符号，需启动一次看 ImportError / AttributeError
3. torch compile / offload 若想保留，需参考 `git show c4f237d0:recipe/dancegrpo/dp_actor.py` 把 verl_0902 的相应逻辑补回去

---

## 7. 如果未来要"严格包含 wxe 改动"的替代方案

本次用 `ours` 是为**保住 verl_0902 所有上游更新**。如果后续判断 wxe 的改动更重要，可切换到：

- **策略 theirs**（`git merge dance-grpo/wxe --allow-unrelated-histories -X theirs`）：冲突时自动取 wxe，会丢失 verl_0902 的 wan22 / torch compile / 上游 verl rebase
- **策略 rebase-then-replay**：对 wxe 22 个 wxe-only commits 生成 patch 集（`git format-patch dance-grpo/cloud..dance-grpo/wxe`），在 verl_0902 上逐个 `git am`（但 patch 大概率会因上下文不匹配而失败，需人工调）

两个替代方案都比本次 `ours` 方案更耗时，且产物可跑性更差，不建议除非有强业务需求。

---

## 8. 验证与后续

### 已做

- [x] 744 个冲突 0 残留（`git diff --name-only --diff-filter=U` 返回空）
- [x] merge commit 已提交：`c4f237d0`
- [x] 分支重命名：`merge_wxe_tmp` → `verl_0902_wxe`

### 未做（使用者负责）

- [ ] 跑 `recipe/dancegrpo/run_dancegrpo.sh`（verl_0902 版）验证合并后 verl_0902 功能正常
- [ ] 按 §6 的顺序逐项 cherry-pick wxe 算法修复
- [ ] 推送到远程：`git push origin verl_0902_wxe`（未自动执行，需用户决定目标 remote 和是否推送）

---

## 9. 关键命令历史

```bash
# 添加 dance-grpo 作为本地 remote 并 fetch
git remote add dance-grpo /Users/wuxuaner/Desktop/teleai/Dance-grpo
git fetch dance-grpo

# 建分支并 merge
git checkout -b merge_wxe_tmp origin/verl_0902
git merge dance-grpo/wxe --allow-unrelated-histories --no-commit --no-ff
# → 744 conflicts

# 分 Tier 批量解决
git diff --name-only --diff-filter=U > /tmp/all_conflicts.txt
# (Tier A 人工列 7 个)
comm -23 <(sort /tmp/all_conflicts.txt) <(sort /tmp/tier_a.txt) > /tmp/tier_b.txt
cat /tmp/tier_b.txt | xargs -I {} git checkout --ours -- "{}"
cat /tmp/tier_b.txt | xargs git add
cat /tmp/tier_a.txt | xargs -I {} git checkout --ours -- "{}"
cat /tmp/tier_a.txt | xargs git add

# 提交与改名
git commit -m "Merge branch 'dance-grpo/wxe' ..."
git branch -m merge_wxe_tmp verl_0902_wxe
```

---

## 10. 运行资源需求（合并后跑 `recipe/dancegrpo/run_dancegrpo*.sh`）

### 10.1 脚本默认配置

4 个启动脚本（`run_dancegrpo.sh` / `run_dancegrpo_joint.sh` / `run_dancegrpo_qwen.sh` / `run_dancegrpo_single.sh`）都是：

```bash
NNODES=${NNODES:-1}            # 节点数，默认 1，支持 env 覆盖
trainer.n_gpus_per_node=8      # 每节点 8 张 GPU
sp_size=1                      # Ulysses 序列并行度（未开）
```

**最小配置**：1 节点 × 8 GPU（共 8 卡）。

### 10.2 模型规格（硬编码在 `run_dancegrpo.sh`）

| 角色 | 模型 | 参数量 |
|---|---|---|
| Actor / Rollout / Ref | `Wan-AI/Wan2.2-T2V-A14B` | **14B** 文生视频 |
| Reward Model | `Qwen2.5-VL-7B-Instruct` | **7B** VL |
| VAE | `Wan2.1_VAE.pth` | — |

训练循环内同时要加载 actor、ref、reward 三份模型权重（+ rollout 的 vllm/sglang 引擎），显存压力主要来自这里。

### 10.3 资源需求估算

| 配置 | 能否跑 | 备注 |
|---|---|---|
| 1 × 8 × H100 80G + `offload=True` | 勉强 | 14B 训练 + 7B reward + rollout，单卡 ~60-75G 占用 |
| 1 × 8 × H100 80G + `offload=False` | ❌ OOM 风险高 | 不开 offload 14B 吃不下 |
| **2 × 8 × 80G（16 卡）**｜**推荐起步** | ✅ | 可关 offload 提速 |
| 4 × 8 = 32 卡 | ✅ 吞吐更好 | 可放大 `rollout_batch` / `mini_batch` |
| A100/H100 **40G** 卡 | ❌ 14B 跑不动 | 14B 必须 80G 卡；40G 需换 1.3B 模型 |

### 10.4 调整节点数 / 模型规模

```bash
# 2 节点 × 8 卡
NNODES=2 bash recipe/dancegrpo/run_dancegrpo.sh

# 用 1.3B 降成本（手改脚本里的 model.path）
actor_rollout_ref.model.path='.../Wan-AI/Wan2.1-T2V-1.3B'
# 1.3B 模型下，1 × 8 × 40G 可跑
```

### 10.5 需要业主自行确认

- 模型权重路径 `'/gfs/space/chatrl/users/wxe/Wan-AI/Wan2.2-T2V-A14B'` 是 wxe 开发机的 GFS 路径，**在新环境要改成实际路径**
- Reward 路径 `'/gfs/space/chatrl/public/models/Qwen2.5-VL-7B-Instruct'` 同上
- `data` 路径、`SpectralVolumeReward` 相关权重同理
