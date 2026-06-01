# DCU grouped_gemm 移植与优化日志

## Stage 0 — 基线调研

- 本地 commit: `5e30741`
- 日期: 2026-05-25
- 目标架构: `gfx938`
- 编译器: `/opt/dtk/bin/aicc`, HIP 6.2.0 / DTK clang 18.0.0
- 构建目标: `tile_example_grouped_gemm`
- 源码入口: `example_hcu/ck_tile/19_grouped_gemm`
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### 当前源码能力

DCU 入口目前只有一个可执行文件 `tile_example_grouped_gemm`，通过 `-prec` 进行运行时 dispatch。

| `-prec` | A/B 类型 | 累加类型 | C 类型 | 默认布局 | 状态 |
|---|---:|---:|---:|---|---|
| `fp16` | fp16 | fp32 | fp32 | A 行主序, B 列主序, C 行主序 | 可构建，默认验证失败 |
| `fp8` | fp8 | fp32 | fp32 | A 行主序, B 列主序, C 行主序 | 构建和验证通过 |
| `int8` | int8 | int32 | int32 | A 行主序, B 列主序, C 行主序 | 构建和验证通过 |

与 `composable_kernel_github/example/15_grouped_gemm` 相比，DCU 示例尚未暴露独立的 fp16/bf16/fp32/int8/int4/splitk/fixed_nk/multiple_d 可执行文件。当前实现使用 MMAC 路径，通过 `PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V4>` 使用 `GemmPipelineAgBgCrCompV4`。

### 基线默认形状性能

测试命令：

```bash
/wksp/ai/composable_kernel_dcu/build_dev/bin/tile_example_grouped_gemm \
  -prec=<type> -warmup=10 -repeat=100 -validate=<0|1> -group_count=8 -kbatch=1
```

默认生成形状：

| Group | M | N | K |
|---:|---:|---:|---:|
| 0 | 256 | 256 | 640 |
| 1 | 512 | 384 | 896 |
| 2 | 768 | 512 | 1152 |
| 3 | 1024 | 640 | 1408 |
| 4 | 1280 | 768 | 1664 |
| 5 | 1536 | 896 | 1920 |
| 6 | 1792 | 1024 | 2176 |
| 7 | 2048 | 1152 | 2432 |

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | Kernel |
|---|---|---:|---:|---:|---|
| fp16 默认 | 失败, 输出 NaN | 0.539217 | 57.8721 TFlops | 158.244 GB/s | `gemm_grouped_fp16_64x128x64_8x8x4_0x0x0_Persistent` |
| fp16 默认 仅性能 | 未运行 | 0.541230 | 57.6569 TFlops | 157.655 GB/s | 同上 |
| fp8 默认 | 正确 | 3.67970 | 8.48047 TFlops | 15.8688 GB/s | `gemm_grouped_fp8_128x128x128_16x16x4_0x0x0_Persistent` |
| fp8 默认 仅性能 | 未运行 | 3.71509 | 8.39969 TFlops | 15.7177 GB/s | 同上 |
| int8 默认 | 正确 | 0.757492 | 41.1960 TFlops | 77.0867 GB/s | `gemm_grouped_int8_32x32x128_16x16x4_0x0x0_Persistent` |
| int8 默认 仅性能 | 未运行 | 0.757446 | 41.1985 TFlops | 77.0914 GB/s | 同上 |

### 资源占用（构建输出）

| 形状系列 | SGPR | VGPR | LDS bytes/block | Occupancy |
|---|---:|---:|---:|---:|
| fp16 persistent V4 | 72 | 208 | 49152 | 1 wave/SIMD |
| int8 persistent V4 | 63–68 | 136–137 | 16384 | 4 waves/SIMD |

### 后续工作

1. 修复 fp16 正确性问题，确认 fp16 性能数据有效。
2. 在参数接口稳定后添加小型的 C ABI 封装。
3. 逐步移植 GitHub 侧示例，优先移植与现有 DCU MMAC pipeline 兼容的用例。

## Stage 1 — C ABI 封装

- 本地 commit: `5e30741` + 工作区修改
- 日期: 2026-05-25
- 修改内容:
  - 新增 `ck_tile_dcu_grouped_gemm_desc`。
  - 新增 `ck_tile_dcu_grouped_gemm_workspace_size`。
  - 新增 `ck_tile_dcu_grouped_gemm_run`。
  - ABI 接受调用方拥有的设备指针、group 数量、dtype 枚举、A/B 布局字符、可选设备 workspace、HIP stream、warmup、repeat，返回平均延迟（毫秒）。
  - 当前 C 布局为行主序，与现有示例路径一致。
- 验证:
  - 在 `gfx938` 上成功重新构建 `tile_example_grouped_gemm`。
  - 重新运行现有 CLI fp8 和 int8 默认验证（GPU 0），均保持正确。
  - 确认构建的二进制文件导出了未修饰的 C 符号 `ck_tile_dcu_grouped_gemm_workspace_size` 和 `ck_tile_dcu_grouped_gemm_run`。

### Stage 1 性能检查

以下使用较短计时循环（`-warmup=3 -repeat=10`），仅用于确认封装修改未导致现有 CLI 路径性能回退。

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 |
|---|---|---:|---:|---:|---:|
| fp8 默认 | 正确 | 3.68878 | 8.4596 TFlops | 15.8298 GB/s |
| int8 默认 | 正确 | 0.757520 | 41.1945 TFlops | 77.0839 GB/s |

## Stage 2 — fp16 输出问题排查

- 本地 base commit: `20dc72c`
- 日期: 2026-05-25
- 修改内容:
  - 将 fp16 grouped_gemm 的 C 输出类型从 `float` 改回 `ck_tile::half_t`，与 `example_hcu/ck_tile/03_gemm/gemm_basic.hpp` 保持一致。
  - 添加编译期守卫，使 fp16 half-output 路径仅实例化 `memory_operation_enum::set`；fp16 SplitK 仍不支持，因为 DTK 无法选择 half 类型的 `atomic_add`。
  - 修复验证阈值计算方式，改为使用参考输出的最大绝对值而非普通最大值。
- 构建: 在 `gfx938` 上成功重新构建 `tile_example_grouped_gemm`。
- 结果:
  - 之前的 fp16 全 NaN 失败变为局部首元素不匹配。
  - fp16 仍不通过验证，尚不能视为支持的/correct 的用例。
  - 相同的首 5 元素 fp16 验证模式在 `tile_example_gemm_basic`（`M=256,N=256,K=640,A=R,B=C,C=R`）中同样出现，说明此问题不限于 grouped_gemm。

### Stage 2 fp16 测量

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | 备注 |
|---|---|---:|---:|---:|---:|---|
| grouped fp16 默认 | 失败 | 0.534531 | 58.3795 TFlops | 130.206 GB/s | 每组前 5 元素错误；无 NaN |
| regular gemm fp16 256x256x640 | 失败 | 0.0332594 | 2.52218 TFlops | 23.6454 GB/s | 相同的首 5 元素模式 |

### Stage 2 结论

fp16 路径已比 Stage 0 更接近可用实现（不再产生 NaN），但正确性仍未解决。下一步 fp16 工作应检查与 `03_gemm` 共享的 CShuffle/host 验证行为，或在找到 NaN 来源并修复后添加单独的 fp16-float-output 路径。

## Stage 3 — int8 64x64 默认候选

- 本地 base commit: `82242df`
- 日期: 2026-05-25
- 修改内容:
  - 新增 `GemmConfigComputeV7<int8>`，使用 `M_Tile=64`, `N_Tile=64`, `K_Tile=128`, `M_Warp=2`, `N_Warp=2`, `K_Warp=1`, `M_Warp_Tile=16`, `N_Warp_Tile=16`, `K_Warp_Tile=32`，persistent 调度和双 LDS 缓冲。
  - CLI int8 默认和 C ABI int8 dispatch 切换到此 64x64 配置。
  - 保留之前的 32x32 int8 配置，可通过 `-config=int8_32x32` 进行比较。
- 构建: 在 `gfx938` 上成功重新构建 `tile_example_grouped_gemm`。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 3 int8 性能

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | 相对 Stage 0 int8 加速 |
|---|---|---:|---:|---:|---:|---:|
| int8 32x32 之前默认 | 正确 | 0.757268 | 41.2081 TFlops | 77.1095 GB/s | +0.03% |
| int8 64x64 候选 | 正确 | 0.505305 | 61.7560 TFlops | 115.559 GB/s | 延迟降低 33.29% |

### Stage 3 资源占用

| 形状系列 | SGPR | VGPR | LDS bytes/block | Occupancy |
|---|---:|---:|---:|---:|
| int8 32x32 之前默认 | 63–68 | 136–137 | 16384 | 4 waves/SIMD |
| int8 64x64 新默认 | 67–68 | 253–256 | 32768 | 2 waves/SIMD |

### Stage 3 结论

更大的 int8 tile 降低了 launch grid 压力并提高了数据复用度，在默认 grouped 形状上足以抵消更高的 VGPR 压力。这是当前已验证 int8 路径的最佳配置，现已成为示例 CLI 和外部 C ABI 的默认配置。

## Stage 4 — bf8 功能路径

- 本地 base commit: `0911828`
- 日期: 2026-05-25
- 修改内容:
  - 为 `ck_tile::bf8_t` 新增 grouped_gemm 类型映射。
  - 新增 `-prec=bf8` CLI dispatch。
  - 新增 C ABI 枚举值 `CK_TILE_DCU_GROUPED_GEMM_BF8` 及 dispatch。
  - 新增 `GemmConfigComputeBf8`，使用 `128x64x128` block tile 和 `32x64x32` warp tile，匹配 DCU bf8 MMAC 路径。
  - 在 `include/ck_tile` 中补齐缺失的 `bf8/bf8` MMAC warp alias 和 `32x64x32_MR2NR4MI1NI1` dispatcher 条目。
- 构建: 在 `gfx938` 上成功重新构建 `tile_example_grouped_gemm`。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 4 bf8 性能

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | Kernel |
|---|---|---:|---:|---:|---:|---|
| bf8 128x64x128 | 正确 | 7.68726 | 4.05940 TFlops | 7.59602 GB/s | `gemm_grouped_bf8_128x64x128_16x16x4_0x0x0_Persistent` |

### Stage 4 结论

bf8 路径现已功能正常并在默认 grouped 形状上通过验证，但性能不具竞争力。应将其视为新移植的能力而非优化后的路径。下一步 bf8 工作应减少资源压力或添加更好的 MMAC warp/block 形状。

## Stage 5 — bf8 128x128 tile 候选

- 本地 base commit: `464e0f1`
- 日期: 2026-05-25
- 修改内容:
  - 新增 `GemmConfigComputeBf8V2`，使用 `M_Tile=128`, `N_Tile=128`, `K_Tile=128`, `K_Warp_Tile=32`。
  - CLI bf8 默认和 C ABI bf8 dispatch 切换到此 128x128 配置。
  - 保留 Stage 4 bf8 配置，可通过 `-config=bf8_128x64` 访问。
- 构建: 在 `gfx938` 上成功重新构建 `tile_example_grouped_gemm`。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 5 bf8 性能

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | 相对 Stage 4 bf8 加速 |
|---|---|---:|---:|---:|---:|---:|
| bf8 128x64 之前默认 | 正确 | 7.68861 | 4.05868 TFlops | 7.59468 GB/s | — |
| bf8 128x128 候选 | 正确 | 4.25201 | 7.33902 TFlops | 13.7329 GB/s | 延迟降低 44.70% |

### Stage 5 结论

128x128 bf8 tile 是当前已验证的 bf8 最佳配置。仍比 fp8 路径慢，下一步 bf8 优化应检查 bf8 MMAC 循环周围的资源使用和 ISA，而非简单地假设 fp8 形状足够。

## Stage 6 — bf8 K64 warp 迭代

- 本地 base commit: `9cc6f3f`
- 日期: 2026-05-26
- 修改内容:
  - 新增 bf8/bf8 MMAC warp alias 和 `32x64x64_MR2NR4MI1NI1` dispatcher 条目。
  - 新增 `GemmConfigComputeBf8K64`，保持 `128x128x128` block tile 不变，将 `K_Warp_Tile` 从 `32` 增加到 `64`。
  - CLI `-prec=bf8` 和 C ABI bf8 dispatch 切换到此 K64 配置。
  - 保留之前 Stage 5 配置，可通过 `-config=bf8_128x128` 访问；Stage 4 仍为 `-config=bf8_128x64`。
- 构建: 在 `gfx938` 上成功重新构建 `tile_example_grouped_gemm`。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 6 bf8 候选结果

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | 相对 Stage 5 bf8 加速 |
|---|---|---:|---:|---:|---:|---:|
| bf8 128x128 K32 之前默认 | 正确 | 4.27152 | 7.30551 TFlops | 13.6702 GB/s | — |
| bf8 128x128 K64 候选 | 正确 | 3.71934 | 8.39011 TFlops | 15.6997 GB/s | 延迟降低 12.93% |

### Stage 6 默认回退检查

将 K64 提升为默认 bf8 路径后，`int8`、`fp8`、`bf8` 重新构建并在一次运行中基准测试：

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 |
|---|---|---:|---:|---:|---:|
| int8 默认 | 正确 | 0.505396 | 61.7448 TFlops | 115.538 GB/s |
| fp8 默认 | 正确 | 3.68659 | 8.46464 TFlops | 15.8392 GB/s |
| bf8 K64 默认 | 正确 | 3.68370 | 8.47128 TFlops | 15.8516 GB/s |

### Stage 6 结论

K64 warp 迭代在不改变 block tile 或验证阈值的情况下提升了 bf8 吞吐量。路径仍有 1 wave/SIMD 的 occupancy 和较高的 LDS 使用量，后续 bf8 工作应比较 ISA 调度和 LDS 读写平衡，而非仅增加 block 大小。

## Stage 7 — 构建迭代速度与移植覆盖范围重置

- 本地 base commit: `cf1b40a`
- 日期: 2026-05-26
- 修改内容:
  - 修改 `script/cmake-ck-dev.sh`，使 `-save-temps` 和 verbose makefile 通过 `CK_SAVE_TEMPS=ON` 和 `CK_VERBOSE_MAKEFILE=ON` 选择启用。
  - grouped_gemm 资源报告改为通过 `-DCK_TILE_GROUPED_GEMM_RESOURCE_REPORT=ON` 选择启用。
  - 新增快速迭代目标: `tile_example_grouped_gemm_fp8_fast`、`tile_example_grouped_gemm_bf8_fast`、`tile_example_grouped_gemm_int8_fast`。
  - 快速目标仅实例化一种 dtype 和默认 A 行主序 / B 列主序布局。完整 `tile_example_grouped_gemm` 目标仍为官方的全 dispatch 目标。
- 构建验证: 干净重新配置成功；`tile_example_grouped_gemm_bf8_fast` 构建并验证通过；完整 `tile_example_grouped_gemm` 仍构建成功。

### Stage 7 构建时间

| 目标 | 范围 | 构建时间 | 备注 |
|---|---|---:|---|
| `tile_example_grouped_gemm` | 所有当前 dtype/layout dispatch 路径 | ~1001–1009 s | 仍然是单个重量级翻译单元；`-j8` 无法并行化一个 `.cpp` 编译 |
| `tile_example_grouped_gemm_bf8_fast` | 仅 bf8 + R/C 布局 | 430 s | bf8 迭代编译延迟降低 57% |

### Stage 7 结论

编译瓶颈在于一个大翻译单元中的模板实例化。快速目标适合迭代，但下一步真正的构建系统优化应将 grouped_gemm 按 dtype/layout 拆分为显式实例化翻译单元，使完整构建可以利用多个编译进程。

## Stage 8 — fp16 host 验证修复

- 本地 base commit: `2f43a4f`
- 日期: 2026-05-26
- 修改内容:
  - 新增 `tile_example_grouped_gemm_fp16_fast` 用于专注 fp16 迭代。
  - 新增 `-debug` CLI 开关用于验证摘要输出。
  - 将 fp16 host 输入生成替换为确定性的原始 half-bit 值，避免 DTK host `_Float16` 转换路径产生 `inf`/`NaN` 测试输入。
  - 新增 fp16 专用的原始 bit CPU 参考/检查路径。检查器手动将 half bit 转换为 float，并使用 tensor descriptor 进行布局感知的索引。
- 构建: 在 `gfx938` 上成功重新构建 `tile_example_grouped_gemm_fp16_fast`。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 8 fp16 正确性

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | GPU |
|---|---|---:|---:|---:|---:|---|
| fp16 单 group 64x128x128 debug | 正确 | 0.019840 | 0.105703 TFlops | 3.30323 GB/s | 0 |
| fp16 默认 8 groups, 短运行 | 正确 | 0.535073 | 58.3204 TFlops | 130.074 GB/s | 0 |

### Stage 8 结论

fp16 grouped_gemm kernel 路径现已针对默认行/列布局和默认 grouped 形状通过验证。之前的 fp16 失败至少部分由 host 侧 `_Float16` 输入生成和验证/参考转换引起，而非 grouped_gemm kernel 本身。下一个移植缺口是 bf16 支持。

## Stage 9 — bf16 功能路径

- 本地 base commit: `a35d076`
- 日期: 2026-05-26
- 修改内容:
  - 新增 `GemmTypeConfig<ck_tile::bf16_t>`，使用 bf16 A/B/C 和 fp32 累加。
  - 新增 `-prec=bf16` CLI dispatch。
  - 新增 C ABI 枚举值 `CK_TILE_DCU_GROUPED_GEMM_BF16` 及 dispatch。
  - 新增 `tile_example_grouped_gemm_bf16_fast`。
  - 补齐缺失的 DCU MMAC warp alias 和 dispatcher 条目 `bf16 16x64x32_MR1NR4MI1NI1`。
  - 为 bf16 输出禁用 SplitK atomic 模式，与 fp16 输出行为一致。
- 构建: 重新配置 `build_dev` 并在 `gfx938` 上成功构建 `tile_example_grouped_gemm_bf16_fast`。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 9 bf16 正确性

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | GPU |
|---|---|---:|---:|---:|---:|---|
| bf16 单 group 64x128x128 debug | 正确 | 0.020000 | 0.104858 TFlops | 3.27680 GB/s | 0 |
| bf16 默认 8 groups, 短运行 | 正确 | 0.534400 | 58.3938 TFlops | 130.238 GB/s | 0 |

### Stage 9 结论

bf16 现已成为默认行/列布局和默认 grouped 形状下的已验证 grouped_gemm dtype。当前复用了 fp16 V4 tile，应将其视为功能性移植步骤而非已调优的 bf16 最佳配置。下一个覆盖缺口是 multiple-D 支持。

## Stage 10 — fp16 multiple-D 功能路径

- 本地 base commit: `55be464`
- 日期: 2026-05-26
- 修改内容:
  - 将 grouped GEMM host/device 参数封装泛化为支持 `NumDTensor`，同时保留现有 0-D 别名。
  - 将 D tensor 指针和 stride 通过 `GroupedGemmKernel` 传递到 `UniversalGemmKernel`。
  - 为现有 `AddAdd` element-wise 路径启用 `CShuffleEpilogue` D tensor 应用。
  - 新增 CLI 选项 `-multiple_d=1`，当前支持 fp16 A/B/C 配两个行主序 D tensor，计算 `C = GEMM(A, B) + D0 + D1`。
  - 新增 fp16 multiple-D CPU checker，考虑 half-output epilogue 舍入并使用宽松的 multiple-D 绝对阈值。
- 构建: 在 `gfx938` 上成功重新构建 `tile_example_grouped_gemm_fp16_fast`。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 10 multiple-D 正确性

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | GPU |
|---|---|---:|---:|---:|---:|---|
| fp16 multiple-D 单 group 64x128x128 debug | 正确 | 0.025760 | 0.0814112 TFlops | 3.81615 GB/s | 0 |
| fp16 multiple-D 默认 8 groups, 短运行 | 正确 | 0.568848 | 54.8575 TFlops | 177.651 GB/s | 0 |

### Stage 10 结论

首个 multiple-D 移植目标（fp16 + 两个 D tensor）现已在默认行/列示例路径上功能正常。当前使用非 persistent grouped launch，因为 persistent tile-loop 参数路径仍为 0-D descriptor 特化。下一步 multiple-D 工作应添加 persistent `NumDTensor` tile-loop 路径，或在完整 0-D 回退构建后扩展 dtype/layout 覆盖。

## Stage 11 — multi-D dtype 扩展

- 本地 base commit: `bfb492c`
- 日期: 2026-05-26
- 修改内容:
  - 将 multiple-D（AddAdd epilogue）支持从 fp16-only 扩展到全部五种 dtype: fp16, bf16, fp8, bf8, int8。
  - C ABI struct 中新增 `num_d_tensors`, `d_ptrs`, `stride_Ds` 字段。
  - `ck_tile_dcu_grouped_gemm_workspace_size` 新增可选 `num_d_tensors` 参数。
  - C ABI dispatch 在 `num_d_tensors == 2` 时构造 multi-D kernel args。
  - 在 `element_wise_operation.hpp` 中为 `bf16_t` 和 `int32_t` 新增 `AddAdd` element-wise 特化。
  - 修复非 persistent grouped_gemm 的 `RunSplitk`，排除 `bf16_t` 的 `atomic_add` 路径（与 `half_t` 一致）。
  - 移除 CLI multi-D dispatch 的 fp16-only 守卫。
- 构建: 全部五个快速目标（`fp16/bf16/fp8/bf8/int8_fast`）在 `gfx938` 上成功构建。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 11 性能（warmup=10, repeat=100）

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | vs 0-D basic |
|---|---:|---:|---:|---:|---:|
| fp16 basic | 正确 | 0.534511 | 58.3816 TFlops | 130.211 GB/s | 基线 |
| fp16 multi-D | 正确 | 0.567390 | 54.9985 TFlops | 178.108 GB/s | +6.2% 延迟 |
| bf16 basic | 正确 | 0.531614 | 58.6998 TFlops | 130.921 GB/s | 基线 |
| bf16 multi-D | 正确 | 0.573051 | 54.4552 TFlops | 176.348 GB/s | +7.8% 延迟 |
| fp8 basic | 正确 | 3.67763 | 8.48525 TFlops | 15.8778 GB/s | 基线 |
| fp8 multi-D | 正确 | 5.10977 | 6.10706 TFlops | 23.7403 GB/s | +38.9% 延迟 |
| bf8 basic | 正确 | 3.67686 | 8.48703 TFlops | 15.8811 GB/s | 基线 |
| bf8 multi-D | 正确 | 5.10996 | 6.10682 TFlops | 23.7393 GB/s | +39.0% 延迟 |
| int8 basic | 正确 | 0.505294 | 61.7574 TFlops | 115.562 GB/s | 基线 |
| int8 multi-D | 正确 | 0.522486 | 59.7253 TFlops | 232.173 GB/s | +3.4% 延迟 |

### Stage 11 结论

Multi-D（C = GEMM(A,B) + D0 + D1，element-wise AddAdd）现已在全部五种 dtype 上功能正常并通过验证。fp8/bf8 multi-D 路径因非 persistent launch 路径和额外内存流量，开销显著（+39%）。int8 multi-D 开销最低（+3.4%）。未来优化应为 multi-D 路径添加 persistent tile-loop 支持。

## Stage 12 — multi-D MultiplyMultiply epilogue

- 本地 base commit: `bfb492c`
- 日期: 2026-05-26
- 修改内容:
  - 在 `element_wise_operation.hpp` 中新增 `MultiplyMultiply` element-wise 操作 struct，包括 `half_t`、`fp16x2_t`、`bf16_t`、`float`、`int32_t` 特化。
  - 新增 `-multiple_d_op` CLI 选项（"add"/"multiply"）。
  - 为 `run_grouped_gemm_multiple_d_example_with_layouts` 新增 `CDEElementWise` 模板参数（默认 `AddAdd` 以保持向后兼容）。
  - 新增 MultiplyMultiply 验证路径（half-output 和非 half-output）。
- 构建: `tile_example_grouped_gemm_fp16_fast` 和 `tile_example_grouped_gemm_bf16_fast` 在 `gfx938` 上成功构建。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 12 性能（warmup=10, repeat=200）

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 |
|---|---:|---:|---:|---:|
| fp16 multi-D AddAdd | 正确 | 0.568477 | 54.8933 TFlops | 177.767 GB/s |
| fp16 multi-D MultiplyMultiply | 正确 | 0.569882 | 54.7581 TFlops | 177.329 GB/s |
| bf16 multi-D MultiplyMultiply | 正确 | 0.572767 | 54.4822 TFlops | 176.436 GB/s |

### Stage 12 结论

MultiplyMultiply epilogue（C = GEMM(A,B) * D0 * D1）现已在 fp16 和 bf16 上功能正常。性能与 AddAdd 几乎相同，符合预期——epilogue 受内存带宽限制，两种操作计算成本相同。C ABI 路径仍硬编码 `AddAdd`——后续 stage 可为 C ABI descriptor 添加 epilogue 操作枚举。

## Stage 13 — Bias 单 D 融合操作（C = GEMM(A,B) + bias）

- 本地 base commit: `bfb492c`
- 日期: 2026-05-26
- 修改内容:
  - 在 `binary_element_wise_operation.hpp` 中为 `element_wise::Add` 新增 `BF16_t` 和 `int32_t` 特化。
  - 新增 `-bias` CLI 选项。
  - 新增 `run_grouped_gemm_bias_example_with_layouts`（~250 行），使用 `GroupedGemmHostArgsImpl<1>` 和 `element_wise::Add`。
  - C ABI 路径处理 `num_d_tensors == 1`。
  - 关键发现: `GroupedGemmHostArgsImpl<NumDTensor>`、`GemmTransKernelArgImpl<NumDTensor>` 和 `CShuffleEpilogue` 均以泛型方式处理 NumDTensor=1——仅 API dispatch 层需要修改。
- 构建: `tile_example_grouped_gemm_fp16_fast` 和 `tile_example_grouped_gemm_bf16_fast` 在 `gfx938` 上成功构建。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 13 性能（warmup=10, repeat=200）

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | vs basic GEMM |
|---:|---:|---:|---:|---:|---:|
| fp16 basic | 正确 | 0.534511 | 58.3816 TFlops | 130.211 GB/s | 基线 |
| fp16 bias | 正确 | 0.558691 | 55.8571 TFlops | 152.820 GB/s | +4.5% 延迟 |
| bf16 basic | 正确 | 0.531614 | 58.6998 TFlops | 130.921 GB/s | 基线 |
| bf16 bias | 正确 | 0.564126 | 55.3227 TFlops | 151.184 GB/s | +6.1% 延迟 |

### Stage 13 结论

Bias 单 D 融合操作（C = GEMM(A,B) + bias）在 fp16 和 bf16 上完全功能正常。相比 basic GEMM 的开销（fp16 +4.5%, bf16 +6.1%）低于 multi-D 开销（~6-8%），因为只需读取一个额外 tensor 而非两个。后续工作: 为 fp8/bf8/int8 dtype 添加 bias 支持，研究 bias 路径的 persistent tile-loop 以匹配 basic GEMM 延迟。

## Stage 14 — 拆分为显式实例化翻译单元

- 本地 base commit: `bfb492c`
- 日期: 2026-05-26
- 修改内容:
  - 创建 `instances/` 子目录，遵循 `17_fused_moe`、`18_moe_quant` 的现有模式。
  - 创建 `instances/grouped_gemm_impl.hpp`——共享内部头文件，包含从 `grouped_gemm.cpp` 提取的所有模板函数定义。
  - 创建 5 个按 dtype 划分的翻译单元: `grouped_gemm_fp16.cpp`、`grouped_gemm_bf16.cpp`、`grouped_gemm_fp8.cpp`、`grouped_gemm_bf8.cpp`、`grouped_gemm_int8.cpp`。
  - 重构 `grouped_gemm.cpp` 从 938 行到 ~160 行，仅保留入口函数和 `main()`。
  - 修复 `json_dump.hpp`（框架）: 为所有 19 个非模板函数添加 `inline` 关键字，避免多翻译单元中的重复符号错误。
- 构建: 完整目标 6 个翻译单元并行编译；全部 5 个快速目标构建成功。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 14 性能（warmup=10, repeat=100）

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 |
|---:|---:|---:|---:|---:|
| fp16 basic | 正确 | 0.534045 | 58.4325 TFlops | 130.325 GB/s |
| bf16 basic | 正确 | 0.531600 | 58.7013 TFlops | 130.924 GB/s |
| fp8 basic | 正确 | 3.699550 | 8.43497 TFlops | 15.7837 GB/s |
| bf8 basic | 正确 | 3.682160 | 8.47481 TFlops | 15.8582 GB/s |
| int8 basic | 正确 | 0.505238 | 61.7642 TFlops | 115.574 GB/s |
| fp16 bias | 正确 | 0.558253 | 55.8987 TFlops | 152.848 GB/s |
| bf16 bias | 正确 | 0.564101 | 55.3193 TFlops | 151.264 GB/s |
| fp16 multi-D AddAdd | 正确 | 0.567509 | 54.9870 TFlops | 178.070 GB/s |
| fp16 multi-D MultiplyMultiply | 正确 | 0.569795 | 54.7664 TFlops | 177.356 GB/s |

### Stage 14 结论

性能与拆分前基线完全一致（在测量噪声范围内），确认零行为回退。完整构建现在并行编译 6 个翻译单元，减少了 wall-clock 编译时间。`instances/` 模式与此代码库中的既定惯例匹配。

## Stage 15 — fp8/bf8 tile 候选筛选

- 本地 base commit: `f909afb`
- 日期: 2026-05-27
- 修改内容:
  - 新增 fp8 候选配置 `-config=fp8_128x64`，使用 `128x64x128` block tile + `K_Warp_Tile=64`。
  - 尝试编译 fp8 K32 候选但失败——`WarpGemmMmacDispatcher<fp8, fp8, float, 32, 64, 32, ...>` 在当前 DCU ck_tile 树中未实现。
  - 默认 fp8 和 bf8 dispatch 保持不变。
- 构建: `tile_example_grouped_gemm_fp8_fast` 和完整目标构建成功。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 15 性能（warmup=10, repeat=100）

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | vs 当前默认 |
|---|---:|---:|---:|---:|---:|
| fp8 默认 128x128/K64 | 正确 | 3.68656 | 8.46470 TFlops | 15.8393 GB/s | 基线 |
| fp8 128x64/K64 候选 | 正确 | 6.52159 | 4.78497 TFlops | 8.95373 GB/s | -43.5% 吞吐量 |
| bf8 默认 128x128/K64 | 正确 | 3.68734 | 8.46290 TFlops | 15.8360 GB/s | 基线 |
| bf8 128x128/K32 | 正确 | 4.26811 | 7.31134 TFlops | 13.6811 GB/s | -13.6% 吞吐量 |
| bf8 128x64/K32 | 正确 | 7.68723 | 4.05941 TFlops | 7.59605 GB/s | -52.0% 吞吐量 |
| int8 默认 回归 | 正确 | 0.505152 | 61.7748 TFlops | 115.594 GB/s | 无回归 |

### Stage 15 结论

没有产生新的最佳配置。对 fp8 和 bf8 而言，当前 `128x128x128` block + `K_Warp_Tile=64` 仍是默认 grouped 形状上的最佳已验证默认。有意义的负面发现: fp8 目前支持 `32x64x64` MMAC warp 路径而非 bf8 独有的 `32x64x32` 路径，将 N tile 减小到 64 会显著降低吞吐量。

## Stage 16 — fp8 K32 MMAC alias 筛选

- 本地 base commit: `dd1c20b`
- 日期: 2026-05-27
- 源码参考: `mmac.xlsx` 列出 gfx938 支持 `__builtin_hcu_mmac_f32_16x16x32_fp8_fp8`、`bf8_bf8`、`fp8_bf8` 和 `bf8_fp8`，以及 `i32_16x16x64_i4/u4`。
- 修改内容:
  - 新增缺失的 fp8/fp8 warp alias `WarpGemmMmacfp8fp8f32_WT32x64x32_MR2NR4MI1NI1` 及对应 dispatcher 条目。
  - 暴露 `-config=fp8_128x128_k32`，复用 `128x128x128` block tile + `K_Warp_Tile=32`。
- 构建: `tile_example_grouped_gemm_fp8_fast` 和完整目标构建成功。
- 基准测试 GPU: `HIP_VISIBLE_DEVICES=0`

### Stage 16 性能（warmup=10, repeat=100）

| 测试用例 | 验证 | 延迟 (ms) | 吞吐量 | 带宽 | vs fp8 默认 |
|---|---:|---:|---:|---:|---:|
| fp8 默认 128x128/K64 | 正确 | 3.71150 | 8.40782 TFlops | 15.7329 GB/s | 基线 |
| fp8 128x128/K32 alias | 正确 | 4.29492 | 7.26570 TFlops | 13.5957 GB/s | -13.6% 吞吐量 |

### Stage 16 结论

fp8 K32 alias 功能有效，但在默认 grouped 形状上不更快，因此默认仍为 `GemmConfigComputeV5`（`128x128x128`, `K_Warp_Tile=64`）。保留 `-config=fp8_128x128_k32` 作为诊断/覆盖路径，因为它证明了 gfx938 fp8 `16x16x32` builtin 在 grouped_gemm 中可用。

## Stage 17 — int4 grouped_gemm 支持与修复

- 本地 base: Stage 16 之后的工作区，尚未提交
- 日期: 2026-05-28
- 快速目标: `tile_example_grouped_gemm_int4_fast`

### 背景

gfx938 支持 `__builtin_hcu_mmac_i32_16x16x64_i4` int4 MMAC builtin，wrapper 位于 `warp_gemm_attribute_mmac_impl.hpp`。`pk_int4_t` 是 packed int4 类型，2 个逻辑 int4 值占用 1 字节（`PackedSize=2`）。

### 最终修复：两个单位不匹配 bug

经过系统性排查，定位到两个相互关联的单位不匹配 bug：

#### 修复 1: SplitKBatchOffset K-split 偏移量（universal_gemm_kernel.hpp）

**问题**: `SplitKBatchOffset::splitted_k` 和 K-split 偏移量（`as_k_split_offset`、`bs_k_split_offset`）使用逻辑 int4 单位计算，但 tensor view 和 `num_loop` 期望物理 pk_int4 单位。

**文件**: [include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp](include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp)（~323–377 行）

**修复**: 新增 `PackedSize` 计算和 `KReadPacked` 变量，将补齐后的逻辑 K 除以 `PackedSize`。所有偏移量和 `splitted_k` 现使用 packed 单位:

```cpp
constexpr index_t PackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
const index_t KReadPacked = __builtin_amdgcn_readfirstlane(KRead / PackedSize);
// KReadPacked 用于 as_k_split_offset、bs_k_split_offset、splitted_k
```

**修复前影响**: `splitted_k` 为 128（逻辑）而非 64（pk_int4）。`num_loop = ceil(128/64) = 2` 而非 `ceil(64/64) = 1`。Tensor view 索引越界，导致越界读取和重复累加。

**修复后**: 对于 K=256，`splitted_k = 64` pk_int4，`num_loop = 1`。脉冲测试 k=0,128 均通过（修复前分别为 out=2 vs ref=1 和 out=0 vs ref=1）。

#### 修复 2: KIterPerWarp 单位不匹配（block_gemm_areg_breg_creg_v1_new.hpp）

**问题**: `KIterPerWarp = KPerBlock / WG::kK` 将 pk_int4 的 KPerBlock（64 pk_int4 = 128 逻辑）除以逻辑 WG::kK（64 逻辑 int4），结果为 1 而非正确的 2。这意味着 block GEMM 每个 block tile 只处理了 128 个逻辑 int4 中的 64 个。

**关键细节**: V4 pipeline（`GemmPipelineAgBgCrCompV4`）使用 `block_gemm_areg_breg_creg_v1_new.hpp` 中的 `BlockGemmARegBRegCRegV1`，**而非**旧版 `block_gemm_areg_breg_creg_v1.hpp`。旧文件仅供 FMHA（flash attention）使用。

**文件**: [include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_new.hpp](include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_new.hpp)（第 74 行）

**修复**:
```cpp
// 修复前:
static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;  // 64/64=1（错误）
// 修复后:
static constexpr index_t KIterPerWarp = KPerBlock * ck_tile::numeric_traits<ADataType>::PackedSize / WarpGemm::kK;  // 64*2/64=2
```

**修复前影响**: 每个 block 只处理加载的 128 逻辑 int4 中的 64 个。模式: 位置 0–63 被处理，64–127 被跳过，128–191 被处理，192–255 被跳过。

**修复后**: 每个 block 在 2 次 warp GEMM 迭代中处理全部 128 个逻辑 int4。所有脉冲位置和 all-ones 测试均通过。

### 生产配置

修复后恢复为类生产的配置，匹配 `GemmConfigComputeV5` 的风格:

```cpp
template <typename PrecType>
struct GemmConfigComputeInt4 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);  // 128 pk_int4 = 256 逻辑
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;
    static constexpr bool Persistent = true;
    static constexpr bool DoubleSmemBuffer = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;
    static constexpr int kBlockPerCu = 1;
};
```

Dispatch 到 `WarpGemmMmacI4I4I32_WT32x64x64_MR2NR4MI1NI1`。

### 新增的 warp dispatcher 条目

为支持诊断和不同 warp 形状，在 [warp_mmac_gemm_dispatcher.hpp](include/ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp) 中新增以下 int4 dispatcher 条目:

| 形状 | 用途 |
|---|---|
| `WT16x16x64_MR1NR1MI1NI1` | 基础 16x16 int4 MMAC tile |
| `WT16x32x64_MR1NR2MI1NI1` | 窄 MR1 诊断 |
| `WT16x64x64_MR1NR4MI1NI1` | MoE 风格 MR1NR4 诊断 |
| `WT16x64x64_MR1NR2MI1NI2` | N interleave 诊断 |
| `WT32x16x64_MR2NR1MI1NI1` | MR2 窄 N 诊断 |
| `WT32x32x64_MR2NR2MI1NI1` | MR2NR2 诊断 |
| `WT32x64x64_MR2NR4MI1NI1` | 生产用 MR2NR4 路径 |
| `WT32x64x64_MR2NR2MI1NI2` | MR2 N-interleave 诊断 |

### 新增的调试开关

```text
-debug=1              启用验证摘要输出
-int4_const=<value>   将 A 和 B 填充为 packed int4 常量值
-int4_const_a=<value> 将 A 填充为常量，B 随机
-int4_b_one_k=<k>     配合 -int4_const_a=1，将 B 在单个逻辑 K 位置设为 1
```

### 验证测试结果

**诊断配置**（M=16, N=64, K_Tile=64 pk_int4）:

| 测试 | K=256 | K=384 | K=512 | K=1024 |
|---|---:|---:|---:|---:|
| all-ones | PASS | PASS | PASS | PASS |
| random | PASS | PASS | PASS | PASS |
| pulse k=0, K/2, K-1 | PASS | PASS | PASS | PASS |

**生产配置**（M=128, N=128, K_Tile=128 pk_int4）:

| 测试 | K=512 | K=768 | K=1024 | K=2048 |
|---|---:|---:|---:|---:|
| all-ones | PASS | PASS | PASS | PASS |
| random | PASS | PASS | PASS | PASS |
| pulse 5 个位置 | PASS | PASS | PASS | PASS |
| M=256 N=256 K=512 all-ones | PASS | — | — | — |

**初步性能**（生产配置, M=N=128, K=1024）:
- 延迟: 0.0402 ms
- 吞吐量: 0.835 TFlops（int4）
- 带宽: 8.15 GB/s

### 修改文件清单

| 文件 | 修改内容 |
|---|---|
| [universal_gemm_kernel.hpp](include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp) | SplitKBatchOffset: KRead/splitted_k/offsets 除以 PackedSize |
| [block_gemm_areg_breg_creg_v1_new.hpp](include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_new.hpp) | KIterPerWarp: KPerBlock 乘以 PackedSize 再除以 WG::kK |
| [grouped_gemm.hpp](example_hcu/ck_tile/19_grouped_gemm/grouped_gemm.hpp) | 恢复 GemmConfigComputeInt4 生产配置 |
| [warp_mmac_gemm_dispatcher.hpp](include/ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp) | 新增 8 个 int4 dispatcher 条目 |

### V4 pipeline 限制: num_loop >= 2

V4 pipeline（`GemmPipelineAgBgCrCompV4`）使用双 LDS 缓冲，需要 `num_loop >= 2`。当 `num_loop == 1`（K 恰好适配一个 K tile）时，pipeline 会触发 `TailNumber::Three` 路径，执行 3 次 block_gemm 操作但只有 1 个有效 tile，导致输出放大 3 倍。

这是 V4 pipeline 的预存限制，非 int4 特有。对于生产配置（`K_Tile = 128` pk_int4 = 256 逻辑），最小 K 为 512 逻辑 int4。对于诊断配置（`K_Tile = 64` pk_int4），最小 K 为 256 逻辑 int4。

### 已知遗留问题

1. `TileGemmShape::KIterPerWarp` 在 [tile_gemm_shape.hpp](include/ck_tile/ops/gemm/tile/tile_gemm_shape.hpp) 中存在相同的单位不匹配（`kK / WarpTile::kK` 用于 `pk_int4_t`），但仅影响 scale tensor 使用的 `KLanePerBlock`。如果 int4 需要 scale 支持则需要修复。

2. V4 pipeline `num_loop == 1` 情况未正确处理（所有类型的预存问题）。K 必须 >= 2 * KPerBlock packed 元素才能得到正确结果。

### Host 侧 packed stride 注意事项

`HostTensor<pk_int4_t>::GetOffsetFromMultiIndex()` 内部会将 descriptor 的原始偏移量除以 `PackedSize`。如果直接将物理 packed stride 传给 host descriptor，行/列 stride 会被再次除以 PackedSize。当前示例代码在 host 侧验证时将物理 stride 乘以 `PackedSize` 来恢复逻辑 stride，kernel args 仍接收原始物理 packed stride。
