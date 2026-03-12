[English](./README.md) | [中文](./README_zh.md)

# CUDA Kernel Benchmark 说明

该目录存放 xLLM 的 CUDA kernel benchmark 可执行程序。当前 benchmark 主要用于
SM120（Blackwell）FP8 CUTLASS GEMM 的配置分析与性能测量。

## 1. 文件说明

- `cutlass_scaled_mm_sm120_benchmark.cpp`
  用于 SM120 FP8 GEMM dispatch 策略的网格搜索 benchmark。它会遍历一组
  `M, N, K` 形状，输出 latency / TFLOPS / 带宽 / wave efficiency，并可选
  保存 CSV 和 JSON 报告。
- `cutlass_scaled_mm_sm120_config_analysis.cpp`
  基于 GTest 的分析程序，用于校验 dispatch 选择逻辑、验证数值正确性，并观察
  配置切换边界附近的性能连续性。
- `CMakeLists.txt`
  benchmark 目标定义文件。这些二进制默认不会构建，只有在
  `BUILD_CUDA_BENCHMARKS=ON` 时才会启用。

## 2. 构建方式

推荐复用仓库现有的 `setup.py` 入口，通过 `CMAKE_ARGS` 打开 benchmark 构建：

```bash
CMAKE_ARGS="-DBUILD_CUDA_BENCHMARKS=ON" python setup.py build test
```

如果你已经使用独立的 CMake 构建目录，也可以直接构建对应目标：

```bash
cmake -S . -B build -DBUILD_CUDA_BENCHMARKS=ON
cmake --build build --target \
  cutlass_scaled_mm_sm120_benchmark \
  cutlass_scaled_mm_sm120_config_analysis -j
```

## 3. 运行前提

- 运行环境必须可用 CUDA。
- 当前 benchmark 面向 SM120（Blackwell）GPU。
- `cutlass_scaled_mm_sm120_benchmark` 在当前 device 的 compute capability
  小于 `12.0` 时会直接退出。
- `cutlass_scaled_mm_sm120_config_analysis` 在不支持的设备上会通过 GTest
  `SKIP` 跳过，因此可以启动但不会实际执行 SM120 用例。

## 4. 可执行程序

### 4.1 `cutlass_scaled_mm_sm120_benchmark`

用途：
测量 SM120 FP8 GEMM dispatch 策略在典型 LLM shape 下的性能表现。

默认测试形状：

- `N=2048, K=11008`
- `N=22016, K=2048`
- `N=2048, K=2048`
- `N=2560, K=2048`

主要参数：

- `--warmup N`：warmup 次数，默认 `10`
- `--iters N`：benchmark 次数，默认 `100`
- `--quick`：只跑关键边界点
- `--m-start N`：`M` 起始值，默认 `1`
- `--m-end N`：`M` 结束值，默认 `4000`
- `--m-step N`：`M` 步长，默认 `32`
- `--no-save`：不保存结果文件
- `--output DIR`：输出目录，默认 `./benchmark_results`
- `--help`：打印帮助信息

示例：

```bash
./cutlass_scaled_mm_sm120_benchmark --quick
./cutlass_scaled_mm_sm120_benchmark --warmup 20 --iters 200 --output ./benchmark_results
```

输出内容：

- 控制台 summary，包括所选配置、latency、TFLOPS、带宽和 wave efficiency。
- 每个 shape 一份 CSV 文件：
  `sm120_fp8_gemm_N<N>_K<K>.csv`
- 一份 JSON 汇总文件：
  `sm120_analysis.json`

### 4.2 `cutlass_scaled_mm_sm120_config_analysis`

用途：
基于 GTest 验证 dispatch 行为，并分析配置边界切换。

典型检查项包括：

- 不同 `M` 区间的配置选择是否符合预期
- 各配置下的数值正确性
- 关键 dispatch 边界附近是否存在明显性能断崖
- 通过 `FullGridSearch` 用例输出完整的网格搜索式分析结果

示例：

```bash
./cutlass_scaled_mm_sm120_config_analysis
./cutlass_scaled_mm_sm120_config_analysis --gtest_filter='*BoundaryPerformance*'
./cutlass_scaled_mm_sm120_config_analysis --gtest_filter='*FullGridSearch'
```

## 5. 说明

- 这两个目标都是普通 `cc_binary`，不是 CTest 用例。打开
  `BUILD_CUDA_BENCHMARKS=ON` 后，它们会随 `all_tests` 一起构建，但不会被
  CTest 自动执行。
- benchmark 输出主要用于 kernel 调优和回归分析，不应直接作为对外性能结论。
- 如果你修改了 `cutlass_w8a8/c3x/` 下的 dispatch policy、tile shape 或边界
  阈值，建议同时重新运行这两个工具，并在提交变更时附上关键结果。
