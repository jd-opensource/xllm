# TileLang CUDA Runtime Wrappers

This directory is reserved for CUDA-side TileLang runtime wrappers.

Compiler-side Python TileLang kernel definitions live under:

- `xllm/xllm/compiler/tilelang/targets/cuda/kernels`

Runtime wrapper code should stay in this directory so the compiler input and
device execution glue remain separated.
