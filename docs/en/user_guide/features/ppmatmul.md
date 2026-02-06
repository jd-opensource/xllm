# PpMatmul Operator Optimization

## Background

In the inference of large models, matrix multiplication accounts for a high proportion and takes a long time. We have optimized the implementation of the matrix multiplication operator.

## Feature Introduction

The PpMatmul operator uses a Tiling strategy to decompose matrix multiplication into multiple smaller matrix multiplication tasks. However, when the number of tiles is small, tasks cannot be evenly distributed across all NPU cores, leading to the tail effect problem, which affects computational efficiency. We optimize the performance of the PpMatmul operator by prefetching memory or redistributing tasks.

## User Interface

### Operator Direct Call API

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

- `a`: Input matrix A.
- `b`: Input matrix B.
- `out`: Output matrix, storing the computation result.

## Performance Effect

For cases with a small number of tiles (e.g., when M is small, corresponding to a small batch size), there is an **18%** performance improvement of the operator compared to before optimization when (TP=4).