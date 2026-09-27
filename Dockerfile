# syntax=docker/dockerfile:1.7
# =============================================================================
# ComfyUI for NVIDIA DGX Spark (Blackwell GB10, ARM64, sm_121a)
# CUDA 13.0.2 + PyTorch cu130 + SageAttention v3 + NVFP4 + Triton
# =============================================================================

# Conservative default — each NVCC job spawns ~8 sub-threads via --threads=8,
# so 8 build jobs × 8 nvcc threads ≈ 64 active compilers. With ~1.5GB/cc1plus
# that peaks around 100GB on Spark's 121GB unified memory.  Earlier 20-wide
# parallelism + parallel FA2 stage OOM-crashed the host — keep this conservative.
ARG BUILD_JOBS=8

# -----------------------------------------------------------------------------
# Stage 1: base — system deps, PyTorch, common tooling
# -----------------------------------------------------------------------------
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS base

ARG BUILD_JOBS
ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

# DGX Spark GB10 = compute capability 12.1a (Blackwell datacenter variant)
# CUDA 13 NVCC supports sm_121 directly; 12.1a unlocks per-SM tensor core paths
ENV TORCH_CUDA_ARCH_LIST="12.1a"
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_INC_PATH=${CUDA_HOME}/include
ENV PATH=${CUDA_HOME}/bin:/usr/lib/ccache:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=20G
ENV CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache
ENV TRITON_PTXAS_PATH=${CUDA_HOME}/bin/ptxas

# System packages — build toolchain, vision/codec libs ComfyUI/custom nodes need
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      git curl wget ca-certificates gnupg vim less tini \
      python3 python3-pip python3-dev python3-venv python3-setuptools \
      build-essential cmake ninja-build pkg-config ccache \
      libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      libgoogle-perftools4 libtcmalloc-minimal4 \
      ffmpeg libavformat-dev libavcodec-dev libswscale-dev libavutil-dev \
      libsndfile1 libopenblas-dev liblapack-dev \
      libjpeg-turbo8-dev libpng-dev libtiff-dev libwebp-dev \
      git-lfs aria2 rsync \
      libnuma-dev numactl htop iotop \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 venv (Ubuntu 24.04 default) — keep deps isolated, allow global install
ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel packaging build

# uv for fast resolver inside venv
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install uv

# -----------------------------------------------------------------------------
# PyTorch + Triton — CUDA 13, ARM64 wheels with sm_121 support
# Pinning torch 2.9.1+cu130 (first stable wheel that ships sm_121 PTX/SASS)
# -----------------------------------------------------------------------------
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.9.1+cu130 \
      torchvision==0.24.1 \
      torchaudio==2.9.1

# Verify torch sees sm_121
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); \
    print('arches:', torch.cuda.get_arch_list())"

# Common scientific stack
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install \
      numpy scipy pillow opencv-python-headless imageio imageio-ffmpeg \
      einops kornia accelerate safetensors transformers tokenizers \
      sentencepiece protobuf psutil pyyaml tqdm rich \
      huggingface-hub hf-transfer hf_xet \
      "diffusers>=0.34" peft \
      aiohttp aiofiles websockets \
      gguf

# Triton from PyPI — 3.6.0 has ARM64 cp312 wheels, even if torch.compile is off
# (we still want triton.language available so custom kernels build cleanly)
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install "triton==3.6.0" || pip install triton || true

# -----------------------------------------------------------------------------
# Stage 2: build SageAttention v3 from source for sm_121a
# Bundled wheels skip Blackwell datacenter parts; we have to compile.
# -----------------------------------------------------------------------------
FROM base AS sageattn-builder

WORKDIR /build
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    git clone --depth=1 https://github.com/thu-ml/SageAttention.git && \
    cd SageAttention && \
    TORCH_CUDA_ARCH_LIST="12.1a" \
    NVCC_APPEND_FLAGS="-gencode=arch=compute_121a,code=sm_121a -gencode=arch=compute_121,code=sm_121" \
    pip wheel --no-build-isolation -w /wheels . && \
    ls -la /wheels

# -----------------------------------------------------------------------------
# Stage 3: final runtime image
# (FlashAttention 2/3 don't support sm_121 yet — SageAttention v3 covers the
# fast attention path on DGX Spark.  PyTorch's built-in scaled_dot_product
# math kernel is the safety-net fallback.)
# -----------------------------------------------------------------------------
FROM base AS runtime

# Copy compiled SageAttention wheel from builder
COPY --from=sageattn-builder /wheels /wheels-sage

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    if ls /wheels-sage/*.whl >/dev/null 2>&1; then \
      pip install /wheels-sage/*.whl; \
    else \
      echo "WARN: SageAttention wheel missing — runtime will fall back to torch SDPA"; \
    fi

# -----------------------------------------------------------------------------
# ComfyUI — bleeding edge master
# -----------------------------------------------------------------------------
ENV COMFY_HOME=/opt/ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFY_HOME}

WORKDIR ${COMFY_HOME}
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Useful comfy ecosystem extras (loaded but not active until referenced)
# Required deps for custom nodes — fail-fast so install issues are caught
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install \
      comfy-cli \
      ftfy regex \
      omegaconf timm einops_exts \
      av \
      scikit-image scikit-learn \
      matplotlib pandas \
      lmdb \
      pynvml nvidia-ml-py \
      ollama

# Optional deps — best-effort install (ARM64 wheels may be missing for some)
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install eva-decord || pip install decord || true
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install onnxruntime-gpu || pip install onnxruntime || true
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install "xformers; platform_machine != 'aarch64'" || true

# -----------------------------------------------------------------------------
# Pre-bundle essential custom nodes (these will be present even on first start;
# user-installed nodes go into the persistent custom_nodes volume)
# -----------------------------------------------------------------------------
ENV BUNDLED_NODES=/opt/bundled_custom_nodes
RUN mkdir -p ${BUNDLED_NODES} && cd ${BUNDLED_NODES} && \
    git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone --depth=1 https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    git clone --depth=1 https://github.com/city96/ComfyUI-GGUF.git && \
    git clone --depth=1 https://github.com/cubiq/ComfyUI_essentials.git && \
    git clone --depth=1 https://github.com/rgthree/rgthree-comfy.git && \
    git clone --depth=1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone --depth=1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    git clone --depth=1 https://github.com/crystian/ComfyUI-Crystools.git && \
    git clone --depth=1 https://github.com/yolain/ComfyUI-Easy-Use.git && \
    git clone --depth=1 https://github.com/ClownsharkBatwing/RES4LYF.git ComfyUI-RES4LYF && \
    git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone --depth=1 https://github.com/stavsap/comfyui-ollama.git ComfyUI-Ollama && \
    git clone --depth=1 https://github.com/Jonseed/ComfyUI-Detail-Daemon.git && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-PromptRelay.git && \
    git clone --depth=1 https://github.com/pixixai/ComfyUI-AlignLayout.git && \
    git clone --depth=1 https://github.com/jlucasmcrell/ComfyUI-H3-Multishot.git && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-SolAttn_triton.git && \
    git clone --depth=1 https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git

# In-tree extension pack: routes the new ComfyUI 0.20+ Workflow-Overview
# "Missing Models -> Download all/Download" buttons through the server-side
# Manager install API so the file lands in the user's workspace volume,
# not in their browser. Critical for remote-accessed Sparks.
COPY aeon-server-side-downloads ${BUNDLED_NODES}/aeon-server-side-downloads

# Install requirements.txt from each bundled node (best-effort)
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    for d in ${BUNDLED_NODES}/*/; do \
      if [ -f "$d/requirements.txt" ]; then \
        echo "Installing deps for $d"; \
        pip install -r "$d/requirements.txt" || echo "skipped $d"; \
      fi; \
    done

# ComfyUI's new in-frontend Manager integration — pip package + --enable-manager
# flag.  The bundled ltdrdata custom node is *also* present (covers older flows),
# but the new frontend dialog ("install missing nodes") talks to this pip pkg.
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --pre comfyui-manager

# -----------------------------------------------------------------------------
# Runtime helpers — model downloader + entrypoint
# -----------------------------------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY download_models.py /usr/local/bin/download_models.py
COPY workflows /opt/default_workflows
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/download_models.py

# -----------------------------------------------------------------------------
# Runtime environment — DGX Spark unified-memory tuning
# -----------------------------------------------------------------------------
# Triton/Inductor often miscompile for sm_121a today — disable torch.compile by default
ENV TORCH_COMPILE_DISABLE=1
ENV TORCHDYNAMO_DISABLE=1

# Grace-Blackwell coherent fabric — don't fight it
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV CUDA_DEVICE_MAX_COPY_CONNECTIONS=4
ENV CUDA_MODULE_LOADING=EAGER
ENV CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
ENV CUBLAS_WORKSPACE_CONFIG=:0:0
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

# HF download tuning
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_DOWNLOAD_TIMEOUT=120
ENV HF_HOME=/workspace/ComfyUI/.cache/huggingface

# ComfyUI runtime hints
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=20
ENV MKL_NUM_THREADS=20

EXPOSE 8188

WORKDIR ${COMFY_HOME}
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
