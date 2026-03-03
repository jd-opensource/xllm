# 新增 TileLang Kernel（NPU）

本文说明如何在以下目录中新增一个 TileLang AscendC kernel：

- `xllm/xllm/core/kernels/npu/tilelang`

文档基于当前构建流程：

- `xllm/xllm/core/kernels/npu/tilelang/CMakeLists.txt`

## 1. 新增 Python codegen 脚本

在以下目录新增 TileLang 脚本：

- `xllm/xllm/core/kernels/python/tilelang/<kernel>.py`

要求：

- 接受 `--output` 参数，并将 lower 后的 AscendC 源码写入该路径。
- 生成 C++ 源码中的入口函数名保持为 `call`（CMake 流程会自动重命名）。
- 将 kernel 的编译期配置通过脚本参数暴露（例如 `--head-dim`、`--rope-dim` 等）。

## 2. 新增 wrapper C++ 接口（如需要）

如果该 kernel 需要对 xLLM 运行时暴露接口，新增：

- `<kernel>_wrapper.h`
- `<kernel>_wrapper.cpp`

典型 wrapper 模式：

1. 声明生成后的符号：

```cpp
extern "C" void XLLM_TL_<KERNEL>_ENTRY(
    uint8_t* in0, uint8_t* in1, uint8_t* out, int n, aclrtStream stream);
```

2. 定义宏兜底：

```cpp
#ifndef XLLM_TL_<KERNEL>_ENTRY
#define XLLM_TL_<KERNEL>_ENTRY call
#endif
```

3. 在 wrapper 中完成 tensor 的 shape/dtype/layout 校验后调用 `XLLM_TL_<KERNEL>_ENTRY(...)`。

## 3. 在 CMake 中注册 codegen + 编译

在 `xllm/xllm/core/kernels/npu/tilelang/CMakeLists.txt` 中新增一次调用：

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

然后把 object 与依赖 target 追加到列表：

```cmake
list(APPEND TILELANG_KERNEL_OBJECTS "${tilelang_foo_KERNEL_OBJ}")
list(APPEND TILELANG_KERNEL_DEP_TARGETS "${tilelang_foo_OBJ_TARGET}")
```

说明：

- `${tilelang_foo_KERNEL_OBJ}` 与 `${tilelang_foo_OBJ_TARGET}` 由 `tilelang_add_ascendc_kernel(...)` 生成。
- helper 函数会把生成源码中的 `call` 重命名为 `${KERNEL_NAME}_call`。

## 4. 为 wrapper 入口符号添加 compile definitions

若新增了 wrapper 源文件，需要通过 compile definitions 导出入口符号：

```cmake
target_compile_definitions(tilelang_kernels PRIVATE
  XLLM_TL_FOO_ENTRY=${tilelang_foo_ENTRY_SYMBOL}
)
```

同时加入 wrapper 校验逻辑需要的静态 shape/配置宏。

## 5. 将 wrapper 源文件加入 `tilelang_kernels`

把 wrapper 文件加入 `cc_library(tilelang_kernels ...)`：

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

## 6. 接入上层 NPU kernel 库（如需要）

如果需要把 API 暴露到全局 NPU ops：

- 在 `xllm/xllm/core/kernels/npu` 下增加对应实现文件。
- 确保 `xllm/xllm/core/kernels/npu/CMakeLists.txt` 依赖 `:tilelang_kernels`（当前已接入）。

## 7. 验证

在容器中执行：

```bash
cd /your_path_to/xllm
python -m py_compile xllm/core/kernels/python/tilelang/<kernel>.py
python setup.py test --test-name <your_test_target> --device a3
```

现有 rope 回归命令：

```bash
python setup.py test --test-name rope_wrapper_test --device a3
```

## 8. 查看 lowering 后的 Ascend-C 源码

可用两种方式查看生成源码：

1. 手动运行 Python codegen（推荐用于调试）

```bash
cd /your_path_to/xllm
python xllm/core/kernels/python/tilelang/rope.py \
  --output /tmp/rope_lowered.cpp \
  --head-dim 576 \
  --rope-dim 64 \
  --skip-ref-check
```

然后直接查看：

```bash
sed -n '1,200p' /tmp/rope_lowered.cpp
```

2. 通过 CMake 构建产物查看（与真实编译链一致）

先执行一次构建/测试后，在 `build/.../generated/...` 下查看：

```bash
cd /your_path_to/xllm
find build -path "*/xllm/core/kernels/npu/tilelang/generated/*/*_kernel.cpp"
```

例如当前 rope 的输出通常是：

- `build/cmake.linux-aarch64-cpython-311/xllm/core/kernels/npu/tilelang/generated/rope/tilelang_rope_kernel.cpp`

注意：

- 手动 codegen 的入口函数名通常是 `call`。
- 走 CMake 流程时会做符号重命名，变为 `${KERNEL_NAME}_call`（例如 `rope_in_place_call`）。

## 9. 常见问题排查

- `undefined reference to ..._call`：
  - 检查 wrapper 宏 `XLLM_TL_<KERNEL>_ENTRY`。
  - 检查 `target_compile_definitions(tilelang_kernels ...)`。
  - 检查 `tilelang_add_ascendc_kernel` 里传入的 `KERNEL_NAME`。
- 生成源码没有重建：
  - 检查 `PY_SCRIPT` 路径和 `CODEGEN_ARGS` 是否变更。
  - 检查 object target 是否追加到了 `TILELANG_KERNEL_DEP_TARGETS`。
- 运行结果异常：
  - 先独立运行 Python 脚本并开启 reference check，确认数值正确后再接入 C++。
