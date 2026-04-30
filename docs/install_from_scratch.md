# Install from scratch

End-to-end recipe for setting up a fresh Linux GPU host (no pre-built env, no
PVC) to run a 4-GPU smoke through `recipe/dancegrpo/run_dancegrpo_*_smoke.sh`.
Targets Python 3.10 + CUDA 12.4 + 4 × H800 80 GB (or H100). Adjust if your
hardware differs.

For a quicker path when you already have a verl-compatible env (e.g. the
`verlai/verl:vllm017.latest` Docker image), see [`INSTALL.md`](../INSTALL.md).

---

## 1. Prerequisites

- Linux + NVIDIA driver ≥ 12.4
- Python 3.10 (`/usr/bin/python3.10` on NGC 24.08 / Ubuntu 22.04)
- 4 × H100 / H800 80 GB or equivalent GPUs visible to the container
- System packages (one-time):
  ```bash
  sudo apt-get install -y libgl1 libglib2.0-0 python3-tk ffmpeg
  ```
  - `libgl1`, `libglib2.0-0` — required by `opencv-python`
  - `python3-tk` — works around an `hpsv2` import that pulls `tkinter`
  - `ffmpeg` — video I/O for the videophy reward

---

## 2. Pinned dependency stack

| Package      | Version          | Notes                                       |
|--------------|------------------|---------------------------------------------|
| python       | 3.10.12          |                                             |
| torch        | 2.6.0+cu124      | install first; vllm and flash-attn need it  |
| flash-attn   | 2.7.4.post1      | install from the prebuilt wheel             |
| vllm         | 0.8.4            | matches torch 2.6                           |
| transformers | 4.57.1           | downgrade after vllm — see §4 gotcha 4      |
| verl         | 0.4.0            | install with `--no-deps`                    |

The full pin file (every transitive dep we used) is at
[`requirements-pinned.txt`](../requirements-pinned.txt).

---

## 3. Install order

```bash
# 1. fresh venv
python3.10 -m venv ~/.venvs/teleboost-py310
source ~/.venvs/teleboost-py310/bin/activate
pip install -U pip

# 2. torch first
pip install torch==2.6.0

# 3. flash-attn from the prebuilt cp310+torch2.6 wheel
#    (download from https://github.com/Dao-AILab/flash-attention/releases)
pip install --no-deps flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 4. vllm (torch already pinned, won't be re-resolved)
pip install vllm==0.8.4

# 5. TeleBoost runtime deps
git clone https://github.com/FortPercent/TeleBoost.git
cd TeleBoost
pip install -r requirements.txt

# 6. upstream verl (no-deps to avoid clobbering torch/vllm/etc.)
pip install --no-deps -r requirements-verl.txt

# 7. TeleBoost itself in editable mode
pip install --no-deps -e .

# 8. extra reward-side packages not in requirements.txt
pip install opencv-python easydict diffusers hpsv2 tensorboard decord

# 9. bring transformers back to a stable version
#    (vllm 0.8.4's resolver may pull a transformers 5.x pre-release that
#    removed AutoModelForVision2Seq)
pip install "transformers==4.57.1"
```

### 3.1 hpsv2 packaging fixes

Two bugs in the published `hpsv2` wheel that you must work around once:

```bash
HPS_DIR=$(python -c "import os, hpsv2; print(os.path.dirname(hpsv2.__file__))")

# 1. The wheel ships without the BPE vocab.
curl -L -o "$HPS_DIR/src/open_clip/bpe_simple_vocab_16e6.txt.gz" \
  https://raw.githubusercontent.com/tgxs002/HPSv2/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz

# 2. factory.py has a stray `from turtle import forward` (dead code) that
#    pulls in tkinter and crashes on systems without it.
sed -i "/from turtle import forward/d" "$HPS_DIR/src/open_clip/factory.py"
```

---

## 4. Model weights

Place the actor + reward model checkpoints anywhere; the smoke scripts read
their paths from env vars (`TRAIN_FILE`, `TEST_FILE`, `CKPTS_DIR`) or accept
Hydra overrides on the command line. Below is the canonical layout used in
the docs.

| Model | Size | Source |
|---|---|---|
| Wan2.1-T2V-1.3B (DiT + T5 + VAE + tokenizer) | 17 GB | [modelscope `Wan-AI/Wan2.1-T2V-1.3B`](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) · [hf `Wan-AI/Wan2.1-T2V-1.3B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| Wan2.1-T2V-14B | partial | [modelscope `Wan-AI/Wan2.1-T2V-14B`](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.2-T2V-A14B (low + high noise sub-models) | partial | [modelscope `Wan-AI/Wan2.2-T2V-A14B`](https://www.modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B) |
| HPS v2.1 reward | 1.84 GB | [hf `xswu/HPSv2`](https://huggingface.co/xswu/HPSv2) → `HPS_v2.1_compressed.pt` |
| VideoCLIP-XL reward | 1.71 GB | [hf `alibaba-pai/VideoCLIP-XL`](https://huggingface.co/alibaba-pai/VideoCLIP-XL) |
| LAION aesthetic head | 4 KB | [LAION aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor) → `sa_0_4_vit_l_14_linear.pth` |
| OpenAI CLIP ViT-L/14 | 890 MB | [OpenAI CLIP](https://github.com/openai/CLIP) — `ViT-L-14.pt` |
| RAFT optical flow | 21 MB | [princeton-vl RAFT](https://github.com/princeton-vl/RAFT) → `models/raft-things.pth` |
| Videophy `videocon_physics` | 25 GB | [hf `videophysics/videocon_physics`](https://huggingface.co/videophysics/videocon_physics) |

Download example (HuggingFace CLI):

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B  --local-dir ./ckpts/Wan2.1-T2V-1.3B
hf download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./ckpts
hf download alibaba-pai/VideoCLIP-XL --local-dir ./ckpts/rewards/VideoCLIP-XL
hf download videophysics/videocon_physics --local-dir ./ckpts/rewards/videocon_physics
```

Wan checkpoints are large; `modelscope download ...` and `hf download ...`
both support resume (write to a `.cache/...` or `.____temp/...` partial file
first, then atomic rename).

---

## 5. Training data preprocess

Every training row must carry:
- `context_path` — umT5-XXL embedding of the positive prompt (skips text-encoder forward at training time);
- `context_null_path` — umT5-XXL embedding of a single shared negative prompt (CFG).

Use the unified, idempotent `prepare_wan_data.py`:

```bash
python data_preprocess/prepare_wan_data.py \
  --input prompts/hard_50.txt \
  --output_dir data/processed/ \
  --wan_model_path /path/to/Wan2.1-T2V-1.3B
```

It accepts `.txt` (one prompt per line) or `.json` (a list of `{caption, ...}`),
loads the umT5 encoder lazily, and produces:

- `data/processed/processed_wan_prompt.json` — list of `{caption, context_path, context_null_path}`
- `data/processed/context_<i>.npy` — per-prompt positive embedding
- `data/processed/context_null.npy` — single shared negative-prompt embedding

Re-runs are safe: per-row encodes are skipped if `context_path` already points
at an existing file, and the negative-prompt encode is skipped if
`context_null.npy` already exists. To use a different negative prompt, pass
`--negative_prompt "..."`.

The dataset loader fails fast if `context_null_path` is missing — without
it, CFG collapses to `(1+scale) * cond` and reward variance vanishes
(grad_norm=0 across every smoke). That used to be a silent footgun.

---

## 6. Smoke

Edit the paths at the top of `recipe/dancegrpo/run_dancegrpo_*_smoke.sh` (or
override via `TRAIN_FILE` / `TEST_FILE` / `CKPTS_DIR` env vars), then:

```bash
source ~/.venvs/teleboost-py310/bin/activate
cd /path/to/TeleBoost

# 4-GPU 1.3B + HPSv2 (smallest)
bash recipe/dancegrpo/run_dancegrpo_1p3B_4gpu_smoke.sh

# 4-GPU 14B Wan2.2 + HPSv2
bash recipe/dancegrpo/run_dancegrpo_single_4gpu_smoke.sh

# 4-GPU sp_size=2 (exercises the Wan Ulysses SP > 1 patches)
bash recipe/dancegrpo/run_dancegrpo_single_4gpu_smoke_sp2.sh

# 8-GPU sp_size=8
bash recipe/dancegrpo/run_dancegrpo_single_8gpu_smoke_sp8.sh
```

Smoke parameters: `n_gpus=4 (or 8), total_steps=2, train_bsz=2 (or 4 at sp=8),
n_resp=2, h=w=256, frames=9, sampling_steps=1`. Not for training quality.

---

## 7. Verification

```bash
source ~/.venvs/teleboost-py310/bin/activate

# 7.1 versions match
python -c "import torch, vllm, flash_attn, transformers; print(torch.__version__, vllm.__version__, flash_attn.__version__, transformers.__version__)"
# expected: 2.6.0+cu124  0.8.4  2.7.4.post1  4.57.1

# 7.2 reward registry has all 5 entries
python -c "from recipe.dancegrpo.reward_models import RewardRegistry; print(RewardRegistry.list_available())"
# expected: ['aesthetic', 'raft', 'videoclip', 'videophy', 'hps']

# 7.3 wan modules import
python -c "from wan.modules.t5 import T5EncoderModel; print('wan t5 ok')"

# 7.4 GPUs visible
python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

# 7.5 INSTALL.md 8-module import smoke (must print 8/8 OK)
python3 - <<'PY'
mods = [
    "recipe.dancegrpo.main_dancegrpo",
    "recipe.dancegrpo.dancegrpo_ray_trainer",
    "recipe.dancegrpo.dancegrpo_fsdp_worker",
    "recipe.dancegrpo.dp_actor",
    "recipe.dancegrpo.unified_reward_worker",
    "teleboost.workers.reward_manager.dancegrpo",
    "teleboost.workers.rollout.diffusion_rollout",
    "teleboost.workers.sharding_manager.diffusion",
]
ok = 0
for m in mods:
    try:
        __import__(m); ok += 1; print(f"OK   {m}")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {e}")
print(f"{ok}/{len(mods)} pass")
PY
```

---

## 8. Gotchas (in roughly the order you'll hit them)

1. **vllm 0.8.4 forces torch 2.6.** If you start from a torch 2.8 / sglang
   environment, `pip install vllm==0.8.4` will downgrade torch — but the
   diffusion sharding manager still imports `verl.third_party.vllm` symbols,
   so installing vllm separately on a torch 2.6 venv is the simplest path.

2. **Don't let pip build flash-attn from source.** No nvcc inside most
   containers, and the build is slow even when nvcc is present. Use the
   prebuilt cp310+torch2.6 wheel from
   <https://github.com/Dao-AILab/flash-attention/releases> with `--no-deps`.

3. **hpsv2 wheel issues.** The published wheel is missing the BPE vocab and
   has a stray `from turtle import forward` line. See §3.1.

4. **vllm 0.8.4 → transformers pre-release.** vllm's resolver may pull a
   transformers 5.x pre-release which removed `AutoModelForVision2Seq`. Force
   `pip install "transformers==4.57.1"` after vllm.

5. **`reward_models/__init__.py` eager-imports every reward.** Even an
   HPS-only smoke triggers the videoclip / videophy / raft / aesthetic
   imports, so `opencv-python` (videoclip needs `cv2`) and the others
   listed in §3 step 8 must all be installed.

6. **Wan + transformers `model_type` warning is harmless.** transformers
   doesn't recognize `model_type="t2v"` and emits a `UserWarning: Failed to
   create processor`. Ignore.

7. **Two parallel `pip install`s on the same venv deadlock.** Both fight for
   `site-packages/` filesystem locks and end up sleeping. Symptom: both pip
   processes show as `Sl` with no IO and no TCP. Always serialize pip on a
   given venv.

8. **`context_null_path` must exist on every training row.** The dataset
   raises a `KeyError` with a fix-it command if a row is missing it. Without
   the negative-prompt embedding, CFG collapses to `(1+scale) * cond`,
   reward variance goes to ~0, and `actor/grad_norm` is exactly 0 — looks
   like the loop runs but the model is not training. Generate via
   `data_preprocess/prepare_wan_data.py` before any real run.

9. **HPSv2 returns `nan` on near-noise videos.** The default smoke runs
   `actor_rollout_ref.sampling_steps=1`, which leaves the rollout output
   essentially as noise. HPSv2 is trained on clean images and produces
   `nan` on such inputs, which propagates to `train/rewards=nan`,
   `train/advantage=nan`, `actor/grad_norm=0.0`. The smoke loop still runs
   end-to-end (good for shape/integration testing), but no learning happens.
   Fix when verifying gradients: `SAMPLING_STEPS=4` (HPS minimum to return
   finite scores) or `SAMPLING_STEPS=10` (production). Joint and Qwen-VL
   rewards return finite values even at `SAMPLING_STEPS=1`.

10. **`init_same_noise=True` collapses reward variance.** The smoke launcher
    sets it for deterministic outputs. With `n_resp_per_prompt>1` it makes
    every response within a prompt start from the same noise; combined with
    a small number of sampling steps the responses end up near-identical
    and GRPO advantage = `(reward - mean) / std` ≈ 0 even when rewards are
    finite. For a real-gradient smoke, override
    `actor_rollout_ref.init_same_noise=False`.
