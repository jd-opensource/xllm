# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

xLLM uses Python setuptools with CMake for C++ components. The build system supports multiple architectures and devices:

### Build Commands

**Standard build:**
```bash
python setup.py build
```

**Build wheel package:**
```bash
python setup.py bdist_wheel
```

**Build with specific device and architecture:**
```bash
python setup.py build --device=a2 --arch=arm
python setup.py build --device=a3 --arch=x86
python setup.py build --device=mlu --arch=arm
```

**Run tests:**
```bash
python setup.py test
```

**Format code:**
```bash
pre-commit run --all-files
```

The main executable is built to `build/xllm/core/server/xllm`.

### Dependencies

Install development dependencies:
```bash
pip install -r cibuild/requirements-dev.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install --upgrade setuptools wheel
```

## Architecture Overview

xLLM is a high-performance LLM inference framework with a service-engine decoupled architecture:

### Core Components

- **`xllm/xllm.cpp`** - Main entry point, handles command line flags and server initialization
- **`xllm/api_service/`** - HTTP API services (chat, completion, embedding, models)
- **`xllm/core/`** - Core inference engine components:
  - **`common/`** - Shared utilities, device memory management, global flags
  - **`distributed_runtime/`** - Distributed serving and PD (Parameter Disaggregation) support
  - **`framework/`** - Execution orchestration and computation graphs
  - **`kernels/`** - NPU kernel adaptations
  - **`layers/`** - Model layer implementations
  - **`runtime/`** - Worker and executor implementations
  - **`scheduler/`** - Batch and PD schedulers
  - **`util/`** - Utility functions
- **`xllm/models/`** - Model implementations (DeepSeek, Qwen, Llama, etc.)
- **`xllm/processors/`** - VLM (Vision-Language Model) preprocessing
- **`xllm/proto/`** - Communication protocols
- **`xllm/server/`** - HTTP server implementation

### Runtime Architecture

The framework creates a Master node (for coordination) and optionally Assistant Masters (for distributed inference). The main flow:

1. Parse command line options into `Options` object
2. Create appropriate Master implementation based on backend type
3. Start API service with HTTP server
4. Handle inference requests through service layer

### Key Features

- **Multiple Backend Support**: LLM, VLM (Vision-Language Models)
- **Device Support**: NPU (A2, A3), MLU acceleration
- **Distributed Inference**: Multi-node support with parameter disaggregation
- **Memory Optimization**: KV cache management, prefix caching
- **Performance Features**: Speculative decoding, chunked prefill, dynamic load balancing

## Development Workflow

### Code Style

The project uses clang-format with Google style (see `.clang-format`). Pre-commit hooks automatically format code:

```bash
pre-commit install  # Run once to install hooks
```

### Testing

Tests are built using CMake's CTest framework:

- C++ tests are located throughout the codebase with `*_test.cpp` naming
- Python tests use pytest in `cibuild/requirements-test.txt`
- Run all tests: `python setup.py test` or `ctest` in build directory

### Environment Setup

For NPU development, the setup.py script automatically configures:
- NPU toolkit paths (`NPU_TOOLKIT_HOME`, `ATB_HOME_PATH`)
- Library paths and Python paths
- Runtime environment variables for optimal performance

## Docker and Deployment

The project provides Docker images for different architectures:

```bash
# A2 x86
docker pull xllm/xllm-ai:xllm-0.6.0-dev-hb-py3.11-oe24.03-lts

# A2 arm  
docker pull xllm/xllm-ai:xllm-0.6.0-dev-hb-py3.11-oe24.03-lts-aarch64

# A3 arm
docker pull xllm/xllm-ai:xllm-0.6.0-dev-hc-py3.11-oe24.03-lts-aarch64
```

## Launch Command

Start xLLM server:
```bash
./build/xllm/core/server/xllm \
    --model=/path/to/your/llm \
    --backend=llm \
    --port=9977 \
    --max_memory_utilization 0.90
```

## Supported Models

- DeepSeek-V3/R1, DeepSeek-R1-Distill-Qwen
- Kimi-k2
- Llama2/3
- MiniCPM-V, MiMo-VL  
- Qwen2/2.5/3/QwQ, Qwen2.5-VL, Qwen3-MoE

## Git and Development

The repository uses git submodules. After cloning:
```bash
git submodule init
git submodule update
```

The project is currently on the `features/OOC` branch with main branch being `main`.