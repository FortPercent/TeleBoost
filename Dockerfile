# Teletron / TeleBoost training image, aligned with megatron-core 0.16.1.
#
# NGC tag is taken from the *official* Megatron-LM repo at the v0.16.1
# tag — `docker/.ngc_version.lts` pins it explicitly:
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.1/docker/.ngc_version.lts
#       -> nvcr.io/nvidia/pytorch:25.09-py3   (LTS, stable testing matrix)
#   `.ngc_version.dev` uses 25.11-py3 (dev / bleeding edge)
#
# Base ships ABI-aligned: torch (NGC build), CUDA, cuDNN, NCCL, cuBLAS,
# transformer_engine, flash-attn 2/3. We only layer Teletron's
# training-side python deps on top.
#
# Build:
#   docker build -t teleboost:mc0.16.1 -f Dockerfile .
#
# Run (note: --shm-size 512G is required for distributed training):
#   docker run -it --gpus all --shm-size 512G \
#              -v /path/to/data:/data \
#              -v /path/to/code:/workspace \
#              teleboost:mc0.16.1

FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive

# ─── system packages ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        zsh curl wget htop ffmpeg \
        inetutils-ping net-tools zip tmux vim \
        cmake git gcc g++ make unzip rsync \
        openssh-server build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

# ─── shell goodies ────────────────────────────────────────────────────
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
 && chsh -s "$(which zsh)" || true \
 && git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions \
        "${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions" \
 && sed -i 's/^ZSH_THEME="robbyrussell"/ZSH_THEME="ys"/' /root/.zshrc \
 && sed -i 's/^plugins=(git)/plugins=(git zsh-autosuggestions z)/' /root/.zshrc

# ─── pip mirror (Tsinghua) ────────────────────────────────────────────
RUN rm -f /etc/pip.conf /etc/xdg/pip/pip.conf /root/.pip/pip.conf /root/.config/pip/pip.conf /usr/pip.conf \
 && pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ─── delete NGC-bundled opencv to avoid version conflicts ─────────────
# NGC ships an opencv variant whose cv2/typing module references newer
# attributes than what teletron's pinned opencv-python==4.10 expects.
# Glob across Python versions (24.10=py3.10, 25.04=py3.10, 25.09=py3.12).
RUN find /usr/local/lib/python3*/dist-packages -maxdepth 2 -name cv2 -type d -exec rm -rf {} + 2>/dev/null \
 && true

# ─── teletron python deps (driven by requirements.txt) ────────────────
# Single source of truth lives in requirements.txt. DO NOT pip-install
# torch / transformer_engine / flash-attn / NVIDIA CUDA libs here —
# NGC's pre-built versions are ABI-aligned with each other; replacing
# any one breaks the chain.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
