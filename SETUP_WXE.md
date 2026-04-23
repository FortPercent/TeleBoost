# Dance-GRPO / TeleBoost 环境与 4 卡 Smoke 准备完整记录

> 目标：在 K8s 容器（4×H100 80GB）里跑通 `recipe/dancegrpo/run_dancegrpo_1p3B_valfirst_4gpu_smoke.sh` 定制版。
> 这份文档把依赖安装顺序、模型来源、预处理步骤、踩过的坑都写清楚，新机器能按这个复现。

---

## 0. 机器与出网

- SSH: `ssh -p 32585 root@116.238.240.2`
- **4 × NVIDIA H100 80GB HBM3**，192 CPU / 2 TB RAM / 20 TB `/user` PVC 卷
- **没有公网直连**，一切外部请求走容器代理：
  ```bash
  export http_proxy=http://10.127.48.4:3128
  export https_proxy=http://10.127.48.4:3128
  ```
  实际出口 IP 在 **香港 (HK)**，`modelscope.cn` / `huggingface.co` / `github.com` 均可达。
- 内部 PyPI 镜像（aliyun proxy）：`http://10.127.48.3:30081/repository/pip/simple/`（pip 自动走）。容器自身 DNS 只解析 K8s `*.svc.cluster.local`，外域名全部由代理解析。

---

## 1. Python 环境

推荐使用新建的：
```bash
source /user/.venvs/teleboost-py310/bin/activate     # symlink -> dancegrpo-vllm084-py310
```
- Python **3.10.12**
- torch **2.6.0+cu124**
- vllm **0.8.4**（和 torch 2.6 配套，项目原 requirements 就是这个版本）
- flash_attn **2.7.4.post1**（cp310+torch2.6 本地 wheel）
- transformers **4.57.1**

完整 pinned 版本见 `requirements-dancegrpo-py310.txt`。

### 1.1 为什么不用原来的 `wxe-teleboost-py312`
py312 那版是 TeleBoost 建的 sglang 路线（torch 2.8 + sglang + flash_attn_3 beta），故意没装 vllm。Dance-grpo 代码里 `verl/workers/sharding_manager/diffusion.py` 顶部硬 import `from verl.third_party.vllm import LLM, parallel_state`，没装 vllm 就炸 `NoneType.startswith`。所以为 Dance-grpo 专门开了这个 py310 环境。

### 1.2 依赖安装顺序（关键）

完整从零建 venv：

```bash
# 1. 建 venv
python3.10 -m venv /user/.venvs/dancegrpo-vllm084-py310
source /user/.venvs/dancegrpo-vllm084-py310/bin/activate
pip install -U pip

# 2. 先装 torch（后面 vllm/flash_attn 都要它）
pip install torch==2.6.0

# 3. 装 flash_attn（用本地 cp310+torch2.6 wheel, 避免从源码编译）
#    wheel 来自: https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/
pip install --no-deps /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 4. 装 vllm（此时 torch 已就位, 不会重拉）
pip install vllm==0.8.4

# 5. 装 Dance-grpo 项目依赖
cd /user/TeleBoost
pip install -r requirements.txt
pip install -e . --no-deps

# 6. 装 reward 链 + wan 模块缺的外围依赖
pip install opencv-python easydict diffusers hpsv2 tensorboard
pip install "transformers==4.57.1"     # vllm 默认带 5.x, 要降到 4.57.1

# 7. hpsv2 两处必须手改的坑
#    7a. 官方 wheel 漏了 BPE vocab
curl -L -o /user/.venvs/dancegrpo-vllm084-py310/lib/python3.10/site-packages/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz \
  https://raw.githubusercontent.com/tgxs002/HPSv2/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz
#    7b. 文件里有条离奇的 `from turtle import forward`（笔误, 废代码）, 带出 tkinter 在系统 python3.10 上炸
sed -i "/from turtle import forward/d" \
  /user/.venvs/dancegrpo-vllm084-py310/lib/python3.10/site-packages/hpsv2/src/open_clip/factory.py
```

---

## 2. 模型权重

**全部统一放在 `/user/TeleBoost/ckpts/`。** 下载命令都写在表下。

| 模型 | 路径 | 大小 | 来源 Link |
|---|---|---|---|
| **Wan2.1-T2V-1.3B** | `/user/TeleBoost/ckpts/Wan2.1-T2V-1.3B/` | 17 GB | [modelscope `Wan-AI/Wan2.1-T2V-1.3B`](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) · [hf `Wan-AI/Wan2.1-T2V-1.3B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| ├ diffusion_pytorch_model.safetensors | | 5.68 GB | (DiT 主权重) |
| ├ models_t5_umt5-xxl-enc-bf16.pth | | 11.36 GB | (umT5-XXL 文本编码器，仅 encoder 部分) |
| ├ Wan2.1_VAE.pth | | 507 MB | (VAE) |
| └ google/umt5-xxl/ | | ~21 MB | (tokenizer: `spiece.model` + `tokenizer.json`) |
| Wan2.1-T2V-14B | `/user/TeleBoost/ckpts/Wan2.1-T2V-14B/` | ⏸ 部分 | [modelscope `Wan-AI/Wan2.1-T2V-14B`](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.2-T2V-A14B | `/user/TeleBoost/ckpts/Wan2.2-T2V-A14B/` | ⏸ 部分 | [modelscope `Wan-AI/Wan2.2-T2V-A14B`](https://www.modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B) |
| **HPS v2.1** | `/user/TeleBoost/ckpts/HPS_v2.1_compressed.pt` | 1.84 GB | [hf `xswu/HPSv2`](https://huggingface.co/xswu/HPSv2) → `HPS_v2.1_compressed.pt` |
| **VideoCLIP-XL** | `/user/TeleBoost/ckpts/rewards/VideoCLIP-XL/` | 1.71 GB | [hf `alibaba-pai/VideoCLIP-XL`](https://huggingface.co/alibaba-pai/VideoCLIP-XL) |
| aesthetic/ViT-L-14.pt | `/user/TeleBoost/ckpts/rewards/aesthetic/ViT-L-14.pt` | 890 MB | [OpenAI CLIP `ViT-L/14`](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2ae0f5b24521ec07/ViT-L-14.pt) |
| aesthetic/head | `/user/TeleBoost/ckpts/rewards/aesthetic/sa_0_4_vit_l_14_linear.pth` | 4 KB | [LAION aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth) |
| raft-things.pth | `/user/TeleBoost/ckpts/rewards/raft/raft-things.pth` | 21 MB | [princeton-vl RAFT checkpoint](https://github.com/princeton-vl/RAFT) → `models/raft-things.pth` |
| videocon_physics | `/user/TeleBoost/ckpts/rewards/videocon_physics/` | ~25 GB | [hf `videophysics/videocon_physics`](https://huggingface.co/videophysics/videocon_physics) |

### 2.1 下载命令

**走 modelscope 的（Wan 系列）：**
```bash
export http_proxy=http://10.127.48.4:3128 https_proxy=http://10.127.48.4:3128
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B   --local_dir /user/TeleBoost/ckpts/Wan2.1-T2V-1.3B
modelscope download --model Wan-AI/Wan2.1-T2V-14B    --local_dir /user/TeleBoost/ckpts/Wan2.1-T2V-14B
modelscope download --model Wan-AI/Wan2.2-T2V-A14B   --local_dir /user/TeleBoost/ckpts/Wan2.2-T2V-A14B
```

**走 HF 的（VideoCLIP-XL / videocon_physics，modelscope 无镜像）：**
```bash
export http_proxy=http://10.127.48.4:3128 https_proxy=http://10.127.48.4:3128
hf download alibaba-pai/VideoCLIP-XL         --local-dir /user/TeleBoost/ckpts/rewards/VideoCLIP-XL
hf download videophysics/videocon_physics    --local-dir /user/TeleBoost/ckpts/rewards/videocon_physics
hf download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir /user/TeleBoost/ckpts    # 或见下面的 rsync 方式
```

**本地已有通过 rsync 上传的（快，不走外网）：**
```bash
rsync -az -e "ssh -p 32585" \
  <local>/HPS_v2.1_compressed.pt \
  <local>/ViT-L-14.pt \
  <local>/sa_0_4_vit_l_14_linear.pth \
  <local>/raft-things.pth \
  root@116.238.240.2:/user/TeleBoost/ckpts/...
```

### 2.2 断点续传

**modelscope**：写入 `<local_dir>/._____temp/<file>`，下完才原子 `rename` 到正式位置。kill 进程后重新跑同一条命令即 resume。

**hf download**：写到 `<local_dir>/.cache/huggingface/download/*.incomplete`，同样 resume 安全。

---

## 3. 训练数据预处理

### 3.1 已生成的 smoke 数据（50 条）
```
/user/TeleBoost/data/1__3B/rl_embeddings/
├── processed_wan_prompt.json           (50 条: caption + context_path + context_null_path)
├── context_000000.npy … context_000049.npy   (umT5 预编码的正 prompt embedding)
└── context_null.npy                    (负 prompt 编码, CFG 用)
```

### 3.2 重新预处理（换 prompt 集合）
```bash
source /user/.venvs/teleboost-py310/bin/activate
# 默认 INPUT_TXT=/user/TeleBoost/prompts/hard_50.txt (50 条)
# 换成 istock_2000.txt（1999 条）:
INPUT_TXT=/user/TeleBoost/prompts/istock_2000.txt \
  bash /user/TeleBoost/data_preprocess/preprocess_wan_rl_embeddings_1p3B.sh
```

**坑**：`preprocess_wan_embeddings_fromlist.py` **不生成 `context_null.npy`**，要另外跑补丁：
```bash
python /tmp/fix_null.py
```
或直接换成 `preprocess_wan_data.py`（自带 null prompt 逻辑）。

可选的 prompt 文件：
```
/user/TeleBoost/prompts/
├── hard_50.txt       (50 条, smoke 默认)
├── istock_2000.txt   (1999 条)
├── istock_5w.txt     (5w 条)
└── mini_test.txt
```

---

## 4. 运行 Smoke

```bash
source /user/.venvs/teleboost-py310/bin/activate
cd /user/TeleBoost
bash recipe/dancegrpo/run_dancegrpo_1p3B_valfirst_4gpu_smoke_wxe.sh
```

脚本里硬编码的路径（相对原模板改的 4 处）：
- `data.train_files` / `data.val_files` → `/user/TeleBoost/data/1__3B/rl_embeddings/processed_wan_prompt.json`
- `actor_rollout_ref.model.path` → `/user/TeleBoost/ckpts/Wan2.1-T2V-1.3B`
- `actor_rollout_ref.model.vae_model_path` → `/user/TeleBoost/ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`
- `reward_model.model.path` → `/user/TeleBoost/ckpts/HPS_v2.1_compressed.pt`

其他 smoke 参数：`n_gpus=4, total_steps=2, train_bsz=2, n_resp=2, h=w=256, frames=9, sampling_steps=1, val_before_train=True`。

### 4.1 Multi-reward (joint) 脚本要覆盖的路径

```bash
reward_model.joint.models.aesthetic.extra_config.clip_model_path=/user/TeleBoost/ckpts/rewards/aesthetic/ViT-L-14.pt
reward_model.joint.models.aesthetic.extra_config.aes_model_path=/user/TeleBoost/ckpts/rewards/aesthetic/sa_0_4_vit_l_14_linear.pth
reward_model.joint.models.raft.model_path=/user/TeleBoost/ckpts/rewards/raft/raft-things.pth
reward_model.joint.models.videoclip.model_path=/user/TeleBoost/ckpts/rewards/VideoCLIP-XL/VideoCLIP-XL.bin
reward_model.joint.models.videophy.model_path=/user/TeleBoost/ckpts/rewards/videocon_physics
```

---

## 5. 踩过的坑（按遇到顺序）

1. **vllm==0.8.4 会把 torch 从 2.8 强降到 2.6**——所以原 TeleBoost 的 py312 venv 不装 vllm（改用 sglang）。但 Dance-grpo 代码里 `verl.workers.sharding_manager.diffusion` 仍依赖 `verl.third_party.vllm` 的符号，没装 vllm 就 `None.startswith` 炸。**只能为 Dance-grpo 专开一个 py310 + torch2.6 + vllm0.8.4 的 venv。**

2. **`flash-attn` 不能让 pip 从源码编译**（容器没 nvcc）。用本地 cp310+torch2.6 wheel 走 `--no-deps` 装。如果 pip 开始建临时目录 `/tmp/pip-install-*/flash-attn_*`，就是走错路径了。

3. **hpsv2 wheel 两个 bug**：
   - 缺 `bpe_simple_vocab_16e6.txt.gz`（BPE 词表）
   - `factory.py` 里有 `from turtle import forward`（废代码），带出 tkinter，系统 py3.10 没装 tkinter 就崩。

4. **`reward_models/__init__.py` 全量 eager import**：即便 smoke 只用 HPS，也会把 videoclip/videophy/raft/aesthetic 的 import 全触发。所以要额外装 `opencv-python`（videoclip 要 `cv2`），其他的靠本地模块满足。

5. **vllm 0.8.4 默认拉 transformers 5.5.4**（pre-release），里面移除了 `AutoModelForVision2Seq`。必须 `pip install "transformers==4.57.1"` 降回去。

6. **hpsv2 把 protobuf 降到 3.20.3**（老 py312 venv 才会有这个问题，因为它强制要求），会破坏 sglang/grpc。但在当前 py310 venv 里 vllm 0.8.4 期望的就是老 protobuf，所以这里反而不用处理。

7. **`wan` 模块缺 `easydict` / `diffusers`**：原 TeleBoost 环境的 freeze 没有这俩（他们当初没跑到 wan 代码），新 venv 里得加。

8. **训练数据 JSON 里的 `context_path` 指向 `/gemini/space/wuxuaner/...`**——那是别的集群的旧路径，不存在。必须重新跑 `preprocess_wan_embeddings_fromlist.py` 生成新 embedding + 改 JSON。

9. **`context_null.npy` 的生成**：`preprocess_wan_embeddings_fromlist.py` 不写 null embedding（字段在 JSON 里空），必须另补。见 `/tmp/fix_null.py`。

10. **数据集 JSON 需要 `context_null_path` 字段**（`rl_dataset.py:265` 读它），`fromlist` 版本默认只写 `context_path`，得手动给每条加。

11. **`trainer.logger="tensorboard"`** 需要单独 `pip install tensorboard`，Dance-grpo 原 requirements 没列。

12. **transformers 不认 Wan 的 `model_type="t2v"`**：只是 `UserWarning: Failed to create processor`，非阻塞。可以忽略。

13. **集群内 pip mirror** 是 `10.127.48.3:30081`（aliyun proxy），下载速度能到 400 MB/s；但外网模型（HF / modelscope）走 `10.127.48.4:3128` HTTP proxy，速度只有 ~9 MB/s，多任务并发时会分摊。

14. **modelscope 没有镜像的模型**（VideoCLIP-XL / videocon_physics），fallback 到 HF `hf download` 经代理。

---

## 6. 文件清单

**脚本（在 `/user/TeleBoost/` 下）：**
- `recipe/dancegrpo/run_dancegrpo_1p3B_valfirst_4gpu_smoke_wxe.sh` — smoke 主脚本（4 路径已改）
- `data_preprocess/preprocess_wan_rl_embeddings_1p3B.sh` — 数据预处理 shell
- `SETUP_WXE.md` — 本文件
- `requirements-dancegrpo-py310.txt` — 完整 pin 依赖

**辅助（在 `/tmp/` 下）：**
- `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` — 本地 flash_attn wheel
- `fix_null.py` — 补 context_null.npy 的 Python 脚本
- `hps_smoke.py` — HPS 独立加载测试脚本（验证权重可用）

**Docker：**
- `docker/Dockerfile.dancegrpo` — overlay, FROM `teleboost/verl-ngc-vllm08:latest`
- `docker/Dockerfile.ngc.vllm0.8` — verl 官方 base (不动)

**venv：**
- `/user/.venvs/teleboost-py310/` — 真身（Python 3.10 + torch 2.6 + vllm 0.8.4）
- `/user/.venvs/wxe-teleboost-py312/` — 老的 TeleBoost sglang 路线环境，保留做参考

---

## 7. 验证清单（装完后按这个自检）

```bash
source /user/.venvs/teleboost-py310/bin/activate

# 7.1 版本对齐
python -c "import torch, vllm, flash_attn, transformers; print(torch.__version__, vllm.__version__, flash_attn.__version__, transformers.__version__)"
# 期望: 2.6.0+cu124  0.8.4  2.7.4.post1  4.57.1

# 7.2 reward 注册表 5 个齐全
python -c "from recipe.dancegrpo.reward_models import RewardRegistry; print(RewardRegistry.list_available())"
# 期望: ['aesthetic', 'raft', 'videoclip', 'videophy', 'hps']

# 7.3 HPS ckpt 独立加载
python /tmp/hps_smoke.py
# 期望看到 'HPS loaded, params: 986109441'

# 7.4 wan 模块 import
python -c "from wan.modules.t5 import T5EncoderModel; print('wan t5 ok')"

# 7.5 4 张 H100 可见
python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
# 期望: 4 个 'NVIDIA H100 80GB HBM3'
```

---

## 8. Git 工作流（本地 ↔ GitHub ↔ 远端容器）

目标：**本地 Mac 改代码 → push 到 GitHub → 远端容器 git pull → 跑测试**。

### 8.0 网络拓扑 / 为什么要绕一下

| 源 | 本地 Mac | 远端容器 |
|---|---|---|
| 内部 gitlab `code.srdcloud.cn` | ✅ | ❌ DNS 不解析, 代理白名单里也没 |
| `github.com` SSH 22 | ✅ | 只能走 HTTP CONNECT 代理隧穿（代理允许 CONNECT :22）|
| `github.com` HTTPS 443 | ✅ | ✅ 走代理 |

**结论**：GitHub 是唯一两边都能到的 git 源。

### 8.1 一次性配置

**前提**：你的 GitHub 账号要有 `<OWNER>/TeleBoost` 的 push 权限（仓库所有者或 collaborator）。

**远端容器** (`/user/TeleBoost`)——要 HTTP 代理 CONNECT 隧穿 SSH：
```bash
# 生成 key
ssh-keygen -t ed25519 -C "teleboost-container" -f ~/.ssh/id_ed25519 -N ""

# 写 ~/.ssh/config, 把 github.com 流量塞进 HTTP 代理隧道
cat >> ~/.ssh/config <<EOF
Host github.com
    User git
    HostName github.com
    ProxyCommand nc -X connect -x 10.127.48.4:3128 %h %p
    StrictHostKeyChecking accept-new
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
EOF
chmod 600 ~/.ssh/config

# 把 ~/.ssh/id_ed25519.pub 内容贴到 GitHub:
#   Settings → SSH and GPG keys → New SSH key

# 测试 + 改 origin 为 SSH URL
ssh -T git@github.com                  # 应回 "Hi <你的账号>!"
cd /user/TeleBoost
git remote set-url origin git@github.com:<OWNER>/TeleBoost.git
```

**本地 Mac**——如果你 Mac 默认 SSH 身份是另一个 GitHub 账号（比如个人号），要用 host 别名隔离：
```bash
# 生成专用 key
ssh-keygen -t ed25519 -f ~/.ssh/id_teleboost -C "mac-teleboost" -N ""

# 配 host 别名 (不影响默认 github.com 的身份)
cat >> ~/.ssh/config <<EOF

Host github-tb
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_teleboost
    IdentitiesOnly yes
EOF

# 把 ~/.ssh/id_teleboost.pub 加到有写权限的 GitHub 账号

# 浅 clone (完整 history 较大, 浅 clone 秒过)
cd ~/Desktop/teleai
git clone --depth=1 --branch=wxe git@github-tb:<OWNER>/TeleBoost.git TeleBoost-gh

# 验证
ssh -T git@github-tb                   # 应回 "Hi <有写权限的账号>!"
cd TeleBoost-gh && git remote -v       # origin 应以 github-tb: 开头
```

> 如果 Mac 默认 SSH 身份就是有写权限的那个账号，跳过别名那套，直接 `git clone git@github.com:<OWNER>/TeleBoost.git TeleBoost-gh` 即可。

### 8.3 日常流程

**本地 Mac**：
```bash
cd ~/Desktop/teleai/TeleBoost-gh
# 写代码
vim recipe/dancegrpo/xxx.sh
git commit -am "xxx"
git push origin wxe
```

**远端容器**：
```bash
ssh -p 32585 root@116.238.240.2
cd /user/TeleBoost
git pull origin wxe
source /user/.venvs/teleboost-py310/bin/activate
bash recipe/dancegrpo/run_dancegrpo_1p3B_valfirst_4gpu_smoke_wxe.sh
```

### 8.4 原来的内部 gitlab 仓库怎么处理

`/Users/wuxuaner/Desktop/teleai/Dance-grpo/` 那份（origin 指内部 gitlab）**保持不动**，继续推 gitlab 做团队内部可见。GitHub 这条线和 gitlab 线**不做自动双向同步**，需要时手动 `git cherry-pick` 或 `git push origin wxe:wxe-github` 等。
