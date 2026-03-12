[English](./README.md) | [中文](./README_zh.md)

# CUDA Kernel Benchmarks

This directory contains CUDA kernel benchmark binaries for xLLM. The current
benchmarks focus on SM120 (Blackwell) FP8 CUTLASS GEMM configuration analysis
and performance measurement.

## 1. Files

- `cutlass_scaled_mm_sm120_benchmark.cpp`
  Grid-search benchmark for the SM120 FP8 GEMM dispatch strategy. It sweeps a
  set of `M, N, K` shapes, reports latency / TFLOPS / bandwidth / wave
  efficiency, and can save CSV and JSON reports.
- `cutlass_scaled_mm_sm120_config_analysis.cpp`
  GTest-based analysis binary for validating dispatch selection, checking
  numerical correctness, and observing performance continuity near
  configuration boundaries.
- `CMakeLists.txt`
  Benchmark target definitions. These binaries are optional and are only built
  when `BUILD_CUDA_BENCHMARKS=ON`.

## 2. Build

The recommended build flow is to enable the benchmark option through
`CMAKE_ARGS` and reuse the standard `setup.py` entry:

```bash
CMAKE_ARGS="-DBUILD_CUDA_BENCHMARKS=ON" python setup.py build test
```

If you already use a standalone CMake build directory, you can also build the
benchmark targets directly:

```bash
cmake -S . -B build -DBUILD_CUDA_BENCHMARKS=ON
cmake --build build --target \
  cutlass_scaled_mm_sm120_benchmark \
  cutlass_scaled_mm_sm120_config_analysis -j
```

## 3. Runtime Requirements

- CUDA must be available.
- The current benchmarks are intended for SM120 (Blackwell) GPUs.
- `cutlass_scaled_mm_sm120_benchmark` exits if the current device compute
  capability is lower than `12.0`.
- `cutlass_scaled_mm_sm120_config_analysis` uses GTest skip for unsupported
  devices, so it can still be launched on other environments without crashing.

## 4. Binaries

### 4.1 `cutlass_scaled_mm_sm120_benchmark`

Purpose:
Measure the SM120 FP8 GEMM dispatch strategy over representative LLM shapes.

Default test shapes:

- `N=2048, K=11008`
- `N=22016, K=2048`
- `N=2048, K=2048`
- `N=2560, K=2048`

Key options:

- `--warmup N`: warmup iterations, default `10`
- `--iters N`: benchmark iterations, default `100`
- `--quick`: run only key boundary points
- `--m-start N`: start of `M` range, default `1`
- `--m-end N`: end of `M` range, default `4000`
- `--m-step N`: step of `M` range, default `32`
- `--no-save`: do not write result files
- `--output DIR`: output directory, default `./benchmark_results`
- `--help`: print usage

Example:

```bash
./cutlass_scaled_mm_sm120_benchmark --quick
./cutlass_scaled_mm_sm120_benchmark --warmup 20 --iters 200 --output ./benchmark_results
```

Output:

- Console summary with selected configuration, latency, TFLOPS, bandwidth, and
  wave efficiency.
- Per-shape CSV files:
  `sm120_fp8_gemm_N<N>_K<K>.csv`
- One JSON summary file:
  `sm120_analysis.json`

### 4.2 `cutlass_scaled_mm_sm120_config_analysis`

Purpose:
Validate dispatch behavior and inspect boundary transitions with a GTest-based
runner.

Typical checks include:

- configuration selection for different `M` ranges
- numerical correctness across configurations
- performance behavior near key dispatch boundaries
- a full grid-search style report through the `FullGridSearch` test case

Examples:

```bash
./cutlass_scaled_mm_sm120_config_analysis
./cutlass_scaled_mm_sm120_config_analysis --gtest_filter='*BoundaryPerformance*'
./cutlass_scaled_mm_sm120_config_analysis --gtest_filter='*FullGridSearch'
```

## 5. Notes

- These targets are plain `cc_binary` executables, not CTest cases. Enabling
  `BUILD_CUDA_BENCHMARKS=ON` makes them build together with `all_tests`, but
  CTest does not run them automatically.
- The benchmark output is intended for kernel tuning and regression analysis,
  not as an end-user performance claim.
- If you update the dispatch policy, tile shape, or boundary thresholds in
  `cutlass_w8a8/c3x/`, rerun both binaries and attach the key output when
  submitting a change.
