# Graph Mode

## 概述

xLLM 支持 Graph Mode，通过预捕获计算图并在后续执行中重放，减少 CPU 开销并提高推理性能。Graph Mode 在不同硬件平台上均有对应实现。

## 功能介绍

为了优化 Host 侧调度性能，图模式通过在 CPU 一次提交大任务后，设备内部流式执行小 kernel，显著降低启动时间和设备气泡。

在 xLLM 引擎中，Graph Mode 实现了以下特性：

### 动态维度参数化
  - 将除 num_tokens 以外的关键动态维度作为整图输入参数，包括 batch_size、kv_seq_lens、q_seq_lens、block_table_size 等，从而提高灵活性。在进行图的内存分配和内核配置时，利用这些动态参数计算实际所需值，例如通过公式 $block\_table\_size = batch\_size \times (max\_seq\_len / block\_size)$ 计算 block_table_size。在图启动阶段，将上述实际参数传入，以确保 kernel 能够使用正确的 stride 访问数据。

### Piecewise Graph
  - 当部分算子不支持 graph 导致整图无法捕获（break graph）时，对 break 之后的各段（piece）分别捕获 graph。这样在无法整图捕获的情况下，仍能尽可能获得 graph mode 的收益，常用于 prefill、chunked_prefill 等场景。

### 多 shape 复用的显存池
  - 为了避免多 shape 使用单独显存 buffer（输入、输出和中间 Tensor）导致浪费，我们采用了可扩张的显存池。多 shape 复用基地址，不同 shape 对池基地址的偏移量（Offset）不同。
  - 更详细的多 shape 复用内存方案（含问题背景、实现与效果）见：[Graph Mode 多 Shape 复用内存技术文档](graph_mode_multi_shape_memory_reuse.md)

## 使用方式

上述功能已在 xLLM 引擎内部实现，对用户透明。通过 gflags 参数 `enable_graph` 开启，默认为 false。在 xLLM 服务启动脚本中设置为 true 即可，示例如下：

```shell
--enable_graph=true
```

## 性能效果

- 开启 Graph Mode 后，在 Qwen3-0.6B 和 Qwen3-1.7B 等模型上，decode 阶段吞吐 **提升约 8%–10%**。

## 模型支持

下表列出目前各模型在 ACLGraph、CudaGraph、MLUGraph 上的支持情况。

| 模型 | ACLGraph | CudaGraph | MLUGraph |
|------|----------|-----------|----------|
| Qwen3/Qwen3-MoE | ✅ | ✅ | ✅ |
| DeepseekV3.2 | ✅ | | |
| GLM4.5/4.6/4.7 | ✅ | | |
| Qwen2.5-VL | | | ✅ |


!!! warning "为新模型添加 Graph Mode 支持"
    需检查计算过程中用到的 kernel 是否实现了动态维度参数化；若未实现，会造成 break graph，可能需要重新实现 kernel。

## 相关文档

- [Graph Mode 多 Shape 复用内存技术文档](graph_mode_multi_shape_memory_reuse.md)
