# RDT2 Docker Image
# 基于 NVIDIA PyTorch 镜像，支持 CUDA 和 Flash Attention

FROM nvcr.io/nvidia/pytorch:24.01-py3

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV MAX_JOBS=4

# 设置 pip 使用清华源
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 升级 pip
RUN pip install --upgrade pip setuptools wheel

# 安装 Python 依赖（分步安装以便缓存）
# 先安装核心依赖（PyTorch 必须从官方源安装 CUDA 版本）
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --trusted-host download.pytorch.org

# 安装 transformers 生态
RUN pip install --no-cache-dir \
    transformers==4.51.3 \
    accelerate==1.8.1 \
    peft==0.17.1 \
    bitsandbytes==0.46.1 \
    deepspeed==0.17.2 \
    safetensors

# 安装 Flash Attention 2
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# 安装其他依赖
RUN pip install --no-cache-dir \
    absl-py==2.1.0 \
    imgaug==0.4.0 \
    numpy==1.26.4 \
    webdataset==1.0.2 \
    zarr==2.18.3 \
    diffusers==0.35.1 \
    timm==1.0.15

# 安装通用工具包
RUN pip install --no-cache-dir \
    av \
    click \
    dill \
    einops \
    filelock \
    imagecodecs \
    jedi \
    lmdb \
    matplotlib \
    moviepy \
    networkx \
    ninja \
    omegaconf \
    opencv-python \
    pillow \
    scikit-image \
    scikit-learn \
    scipy \
    threadpoolctl \
    tqdm \
    wandb

# 安装 vllm（用于高效推理）
RUN pip install --no-cache-dir vllm

# 安装机器人部署相关依赖（可选）
RUN pip install --no-cache-dir \
    pynput \
    atomics \
    minimalmodbus \
    zerorpc \
    || true

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends ssh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
        
# 设置成docker中已有的python的版本
ENV PYTHON_VERSION=py3.10.12 \
    BASE_ENV_NAME=py3.10.12 \
    JUPYTER_ENV_NAME=py3.10.12

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn\
    ipython \
    jupyterlab>='4.1.0,<5.0.0a0' \
    jupyter \
    nbclassic \
    notebook

# jupyterlab extentions(可选)
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn\
    jupyterlab-lsp \
    anywidget \
    kaleido \
    pyviz_comms \
    lckr_jupyterlab_variableinspector \
    jupyterlab-spreadsheet-editor \
    jupyterlab-spreadsheet \
    jupyterlabcodetoc

# 设置工作目录
WORKDIR /workspace

# 设置 HuggingFace 缓存目录
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# 默认命令
CMD ["/bin/bash"]
