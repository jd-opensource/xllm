# Add a New TileLang Kernel (NPU)

This note describes how to add another TileLang AscendC kernel under:

- `xllm/xllm/core/kernels/npu/tilelang`

It is based on the current build flow in:

- `xllm/xllm/core/kernels/npu/tilelang/CMakeLists.txt`

## 1. Add Python codegen script

Create a new TileLang script in:

- `xllm/xllm/core/kernels/python/tilelang/<kernel>.py`

Requirements:

- Accept `--output` argument and write lowered AscendC source to that file.
- Keep entry function name as `call` in generated C++ source (the CMake flow renames it automatically).
- Keep all kernel compile-time options exposed as script CLI args (example: `--head-dim`, `--rope-dim`, etc.).

## 2. Add wrapper C++ API (if needed)

If this kernel is exposed to xLLM runtime, add:

- `<kernel>_wrapper.h`
- `<kernel>_wrapper.cpp`

Typical wrapper pattern:

1. Declare generated symbol:

```cpp
extern "C" void XLLM_TL_<KERNEL>_ENTRY(
    uint8_t* in0, uint8_t* in1, uint8_t* out, int n, aclrtStream stream);
```

2. Define a macro fallback:

```cpp
#ifndef XLLM_TL_<KERNEL>_ENTRY
#define XLLM_TL_<KERNEL>_ENTRY call
#endif
```

3. Validate tensor shape/dtype/layout and invoke `XLLM_TL_<KERNEL>_ENTRY(...)`.

## 3. Register kernel codegen+compile in CMake

In `xllm/xllm/core/kernels/npu/tilelang/CMakeLists.txt`, add one more call:

```cmake
set(TILELANG_FOO_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/foo")
set(TILELANG_FOO_KERNEL_NAME "foo_in_place")

tilelang_add_ascendc_kernel(
  TARGET tilelang_foo
  PY_SCRIPT "${TILELANG_PY_DIR}/foo.py"
  BUILD_DIR "${TILELANG_FOO_BUILD_DIR}"
  KERNEL_NAME "${TILELANG_FOO_KERNEL_NAME}"
  CODEGEN_ARGS
    --arg0 "..."
    --arg1 "..."
)
```

Then append object and dependency target:

```cmake
list(APPEND TILELANG_KERNEL_OBJECTS "${tilelang_foo_KERNEL_OBJ}")
list(APPEND TILELANG_KERNEL_DEP_TARGETS "${tilelang_foo_OBJ_TARGET}")
```

Notes:

- `${tilelang_foo_KERNEL_OBJ}` and `${tilelang_foo_OBJ_TARGET}` are produced by `tilelang_add_ascendc_kernel(...)`.
- The helper function renames `call` to `${KERNEL_NAME}_call` in generated C++ source.

## 4. Wire compile definitions for wrapper entry symbol

If you add a new wrapper translation unit, export its entry symbol through compile definitions:

```cmake
target_compile_definitions(tilelang_kernels PRIVATE
  XLLM_TL_FOO_ENTRY=${tilelang_foo_ENTRY_SYMBOL}
)
```

Also add any static shape/config macros needed by wrapper checks.

## 5. Add wrapper sources into `tilelang_kernels`

Append wrapper files to `cc_library(tilelang_kernels ...)`:

```cmake
cc_library(
  NAME
    tilelang_kernels
  HDRS
    rope_wrapper.h
    foo_wrapper.h
  SRCS
    rope_wrapper.cpp
    foo_wrapper.cpp
    ${TILELANG_KERNEL_OBJECTS}
  DEPS
    torch
    torch_npu
)
```

## 6. Hook into upper NPU kernel library (if needed)

If your API should be visible in global NPU ops:

- Add corresponding implementation file in `xllm/xllm/core/kernels/npu`.
- Ensure `xllm/xllm/core/kernels/npu/CMakeLists.txt` depends on `:tilelang_kernels` (already true now).

## 7. Validation

Run inside container:

```bash
cd /your_path_to/xllm
python -m py_compile xllm/core/kernels/python/tilelang/<kernel>.py
python setup.py test --test-name <your_test_target> --device a3
```

For existing rope regression:

```bash
python setup.py test --test-name rope_wrapper_test --device a3
```

## 8. View Lowered Ascend-C Source

You can inspect generated source in two ways:

1. Run Python codegen directly (recommended for debugging)

```bash
cd /your_path_to/xllm
python xllm/core/kernels/python/tilelang/rope.py \
  --output /tmp/rope_lowered.cpp \
  --head-dim 576 \
  --rope-dim 64 \
  --skip-ref-check
```

Then inspect the file:

```bash
sed -n '1,200p' /tmp/rope_lowered.cpp
```

2. Inspect CMake build artifacts (matches real compile pipeline)

After one build/test run, locate generated files under `build/.../generated/...`:

```bash
cd /your_path_to/xllm
find build -path "*/xllm/core/kernels/npu/tilelang/generated/*/*_kernel.cpp"
```

For current rope flow, a common output is:

- `build/cmake.linux-aarch64-cpython-311/xllm/core/kernels/npu/tilelang/generated/rope/tilelang_rope_kernel.cpp`

Notes:

- Direct Python codegen typically emits entry symbol `call`.
- CMake flow renames entry symbol to `${KERNEL_NAME}_call` (for example `rope_in_place_call`).

## 9. Troubleshooting

- `undefined reference to ..._call`:
  - Check wrapper macro `XLLM_TL_<KERNEL>_ENTRY`.
  - Check `target_compile_definitions(tilelang_kernels ...)`.
  - Check `KERNEL_NAME` passed to `tilelang_add_ascendc_kernel`.
- Generated source not rebuilt:
  - Confirm `PY_SCRIPT` path and `CODEGEN_ARGS` changed.
  - Confirm object target is appended to `TILELANG_KERNEL_DEP_TARGETS`.
- Wrong runtime behavior:
  - Re-run Python script standalone with ref-check enabled before C++ integration.
