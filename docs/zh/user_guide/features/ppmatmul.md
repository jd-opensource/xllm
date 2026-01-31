# PpMatmul 算子优化

## 背景

针对大模型推理中矩阵乘法占比高、耗时长的问题，优化了矩阵乘法算子的实现。

## 功能介绍

PpMatmul 算子使用 Tiling 切分策略，将矩阵乘法分解为多个小的矩阵乘法任务。然而当 tile 数量较小时任务无法被均匀分配到所有 npu 核心上，导致 tail effect 问题，影响计算效率。我们通过预取内存或重新划分任务的方式，优化 PpMatmul 算子的性能。

## 用户接口

### 算子直调 API

```cpp
aclnnStatus aclnnPpMatmulOptGetWorkspaceSize(
    const aclTensor *a,
    const aclTensor *b,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnPpMatmulOpt(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

- `a`: 输入矩阵 A。
- `b`: 输入矩阵 B。
- `out`: 输出矩阵，存储计算结果。

## 性能效果

对于 tile 数量较小的情况（例如 M 较小，对应于 batch size 较小的情况），在（TP=4）时，算子较优化前有 **18%** 的性能提升。