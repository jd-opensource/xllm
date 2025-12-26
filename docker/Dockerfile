
ARG CUDA_VERSION=12.8.0
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

FROM ${BASE_IMAGE}


ARG CUDA_VERSION
ARG PYTHON_VERSION=3.11
ARG CMAKE_VERSION=3.27.9

ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG TORCHAUDIO_VERSION=2.7.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0 12.0"

ARG FLASHINFER_VERSION=0.3.1
ARG FLASHINFER_CUDA_ARCH_LIST="8.0 8.9 9.0a 10.0a 12.0a"
ARG YALANTINGLIBS_VERSION=0.5.5

ARG GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"

# Optional: use proxy for network access
ARG http_proxy=
ARG https_proxy=
# Optional: use pypi mirror (e.g. https://pypi.tuna.tsinghua.edu.cn/simple)
ARG UV_DEFAULT_INDEX=
# Optional: accelerate github https clone (e.g. https://gh-proxy.com/)
ARG GIT_HTTPS_MIRROR_PREFIX=
# Optional: use rustup proxy (example: https://rsproxy.cn)
ARG RUSTUP_DIST_SERVER=
ARG RUSTUP_UPDATE_ROOT=


SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /workspace

ENV CUDA_VERSION=${CUDA_VERSION}
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV CMAKE_VERSION=${CMAKE_VERSION}

ENV TORCH_VERSION=${TORCH_VERSION}
ENV TORCHVISION_VERSION=${TORCHVISION_VERSION}
ENV TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}


ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 
ENV PYTHONDONTWRITEBYTECODE=1

# # Install system dependencies including build tools 
RUN set -eux; \
    echo "tzdata tzdata/Areas select Asia" | debconf-set-selections; \
    echo "tzdata tzdata/Zones/Asia select Shanghai" | debconf-set-selections; \
    apt-get update -y && apt-get install -y --no-install-recommends \
      ca-certificates gdb wget curl vim git zip bison build-essential \
      libx11-dev libxft-dev libxext-dev \
      autoconf automake libtool pkg-config \
      libxi-dev libxtst-dev \
      libwayland-dev libxrandr-dev libxinerama-dev libxcursor-dev libepoxy-dev libgtk-3-dev \
    libnuma-dev libibverbs-dev nasm software-properties-common; \
    add-apt-repository ppa:deadsnakes/ppa -y; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    libibverbs-dev; \
    rm -rf /var/lib/apt/lists/*; \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1; \
    update-alternatives --set python /usr/bin/python${PYTHON_VERSION}; \
    ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config; \
    curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION}; \
    python --version && pip --version; \
    apt-get clean


# UV envs
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX}
# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy
# Omit development dependencies
ENV UV_NO_DEV=1
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

ENV PIP_INDEX_URL=${PIP_INDEX_URL}
# Install python dependencies
RUN set -eux; \
    pip install \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"; \
    pip install uv ninja "numpy<2" nvshmem4py-cu12 nvidia-nvshmem-cu12 \
    aiohttp requests tqdm transformers

ENV NVSHMEM_HOME=/usr/local

# Install cmake
RUN set -eux; \
    url="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"; \
    curl -fL --retry 3 --retry-delay 2 -o /tmp/cmake.tar.gz "${url}"; \
    mkdir -p /opt/cmake; \
    tar -xzf /tmp/cmake.tar.gz -C /opt/cmake --strip-components=1; \
    rm -f /tmp/cmake.tar.gz; \
    ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake; \
    cmake --version

# Install Rust (minimal profile)
ENV RUSTUP_DIST_SERVER=${RUSTUP_DIST_SERVER}
ENV RUSTUP_UPDATE_ROOT=${RUSTUP_UPDATE_ROOT}
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
ENV PATH=${CARGO_HOME}/bin:$PATH
RUN set -eux; \
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable; \
    rustup --version; cargo --version; rustc --version

# Optional: use https mirror for git clone if provided
ENV GIT_HTTPS_MIRROR_PREFIX=${GIT_HTTPS_MIRROR_PREFIX}
RUN if [[ -n "${GIT_HTTPS_MIRROR_PREFIX}" ]]; then \
      git config --global url."${GIT_HTTPS_MIRROR_PREFIX}".insteadOf https://; \
    fi

# Install yalantinglibs
ENV YALANTINGLIBS_VERSION=${YALANTINGLIBS_VERSION}
RUN set -eux; \
    git clone --depth 1 --branch "${YALANTINGLIBS_VERSION}" https://github.com/alibaba/yalantinglibs.git /workspace/yalantinglibs; \
    cmake -S /workspace/yalantinglibs -B /workspace/yalantinglibs/build \
      -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF \
      -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"; \
    cmake --build /workspace/yalantinglibs/build -j"$(nproc)"; \
    cmake --install /workspace/yalantinglibs/build; \
    rm -rf /workspace/yalantinglibs

# Install tvm-ffi
RUN set -eux; \
    git clone --depth 1 --recurse-submodules https://github.com/apache/tvm-ffi.git /workspace/tvm-ffi; \
    git -C /workspace/tvm-ffi submodule update --init --recursive; \
    cmake -S /workspace/tvm-ffi -B /workspace/tvm-ffi/build_cpp -DCMAKE_BUILD_TYPE=RelWithDebInfo; \
    cmake --build /workspace/tvm-ffi/build_cpp --parallel --config RelWithDebInfo --target tvm_ffi_shared; \
    cmake --install /workspace/tvm-ffi/build_cpp --config RelWithDebInfo; \
    rm -rf /workspace/tvm-ffi

# Install flashinfer + AOT compile
ENV FLASHINFER_VERSION=${FLASHINFER_VERSION}
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
RUN set -eux; \
    git clone --depth 1 --branch "v${FLASHINFER_VERSION}" --recurse-submodules https://github.com/flashinfer-ai/flashinfer.git /workspace/flashinfer; \
    cd /workspace/flashinfer; \
    uv venv --python ${PYTHON_VERSION} && source .venv/bin/activate; \
    uv pip install -v .; \
    uv pip install "cuda-python==${CUDA_VERSION}" setuptools pillow "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" --index-url "${TORCH_INDEX_URL}"; \
    python -m flashinfer.aot; \
    deactivate; \
    mv /workspace/flashinfer/aot-ops /flashinfer_ops; \
    rm -rf /workspace/flashinfer

ENV FLASHINFER_OPS_PATH=/flashinfer_ops
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install VCPKG
RUN set -eux; \
    git clone --depth 1  https://github.com/microsoft/vcpkg.git /vcpkg

ENV VCPKG_ROOT=/vcpkg

# Cleanup
RUN if [[ -n "${GIT_HTTPS_MIRROR_PREFIX}" ]]; then \
      git config --global --unset-all url."${GIT_HTTPS_MIRROR_PREFIX}".insteadOf 2>/dev/null || true; \
    fi; \
    pip cache purge; \
    uv cache clean

USER root
CMD ["/bin/bash"]