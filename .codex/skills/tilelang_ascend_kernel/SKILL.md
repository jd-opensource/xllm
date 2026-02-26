---
name: tilelang-ascend-kernel
description: Use when the user wants to add, modify, debug, or review an xLLM TileLang Ascend kernel or specialization, including Python kernel definitions, generated Ascend-C source, runtime wrapper dispatch, TileLang CMake wiring, and NPU tests.
---

# TileLang Ascend Kernel

## When to use

Use this skill when the task involves any of the following in the xLLM repo:

- `xllm/compiler/tilelang/targets/ascend/kernels/*.py`
- `xllm/core/kernels/npu/tilelang/*_wrapper.cpp`
- `xllm/core/kernels/npu/tilelang/CMakeLists.txt`
- generated TileLang artifacts such as `manifest.json`, `registry.inc`, or specialization `.cpp`

Run build and test commands inside the NPU container.

## Choose the path first

- New `kernel`
  - add a new Python kernel file
  - add a new wrapper entry
  - register it in CMake
- New `specialization`
  - update an existing kernel's `SPECIALIZATIONS`
  - keep wrapper runtime specialization construction aligned with `DISPATCH_SCHEMA`

## New kernel

Follow this order:

1. Implement `build_<kernel>_kernel(...)`
2. Implement `generate_source(...)`
3. Declare `DISPATCH_SCHEMA` and `SPECIALIZATIONS`
4. Run TileLang compilation once and inspect `registry.inc`
5. Add or update `<kernel>_wrapper.cpp`
6. Register the kernel in `xllm/core/kernels/npu/tilelang/CMakeLists.txt` with:
   - `tilelang_register_runtime_kernel(NAME <kernel> WRAPPER_SRCS <srcs...>)`

For wrapper work:

- handwrite tensor checks, layout transforms, and `build_runtime_specialization(...)`
- use generated `make_<kernel>_specialization(...)` and `find_<kernel>_kernel_entry(...)`
- do not handwrite kernel-specific specialization structs or kernel fn typedefs

## New specialization

Use this path when the kernel logic and wrapper ABI stay the same.

1. Update the existing kernel's `SPECIALIZATIONS`
2. Confirm every runtime dispatch field still matches `DISPATCH_SCHEMA`
3. Re-run TileLang compilation
4. Check that `registry.inc` contains the new entry
5. Check that the wrapper's `build_runtime_specialization(...)` still constructs matching values

## Debug generated Ascend-C

When the task is to inspect codegen or compare two kernel implementations, use an isolated output root:

```bash
python xllm/compiler/tilelang_launcher.py compile-kernels \
  --target ascend \
  --device a3 \
  --output-root /tmp/tilelang_debug \
  --kernels <kernel> \
  --force
```

Then inspect:

- `/tmp/tilelang_debug/targets/ascend/<kernel>/<variant_key>/<kernel>_<variant_key>_kernel.cpp`
- `/tmp/tilelang_debug/targets/ascend/<kernel>/registry.inc`
- `/tmp/tilelang_debug/targets/ascend/<kernel>/manifest.json`

Use this path before changing wrapper code when you need to understand generated symbols, field order, or ABI.

## Validate

Prefer the narrowest command first:

- `python -m py_compile xllm/compiler/tilelang/targets/ascend/kernels/<kernel>.py`
- `python xllm/compiler/tilelang_launcher.py compile-kernels --target ascend --device a3 --output-root /tmp/tilelang_debug --kernels <kernel> --force`
- `python setup.py test --test-name <wrapper_test_target> --device a3`

## References

Read the canonical development note for mechanism details:

- `docs/en/dev_guide/tilelang_ascend_kernel_dev.md`

Use the current rope implementation as the concrete template:

- `xllm/compiler/tilelang/targets/ascend/kernels/rope.py`
- `xllm/core/kernels/npu/tilelang/rope_wrapper.cpp`
- `xllm/core/kernels/npu/tilelang/CMakeLists.txt`

Treat `rope` as a concrete example, not as a restriction on the skill itself.
