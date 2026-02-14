# Graph Mode

## Overview

xLLM supports Graph Mode: computation graphs are pre-captured and replayed in subsequent runs to reduce CPU overhead and improve inference performance. Graph Mode has corresponding implementations on different hardware platforms.

## Feature Description

To optimize Host-side scheduling, graph mode submits a large task from the CPU once and then executes small kernels in a streaming manner on the device, significantly reducing startup time and device bubbles.

In the xLLM engine, Graph Mode provides the following:

### Dynamic Shape Parameterization
  - Key dynamic dimensions other than num_tokens are treated as whole-graph input parameters, including batch_size, kv_seq_lens, q_seq_lens, block_table_size, and the like, for flexibility. During memory allocation and kernel configuration, these parameters are used to compute required values—for example, block_table_size via $block\_table\_size = batch\_size \times (max\_seq\_len / block\_size)$. At graph launch, the actual values of these parameters are passed so kernels use the correct strides to access data.

### Piecewise Graph
  - When some operators do not support graph capture and thus break the full graph, each segment (piece) after the break is captured as a separate graph. This maximizes graph-mode benefits even when the full graph cannot be captured, and is commonly used for prefill and chunked prefill.

### Multi-Shape Reusable Memory Pool
  - To avoid waste from separate memory buffers (input, output, and intermediate tensors) per shape, we use an expandable memory pool. Multiple shapes share the pool base address, with different shapes using different offsets from that base.
  - For a detailed multi-shape memory reuse design (background, implementation, and results), see: [Graph Mode Multi-Shape Memory Reuse Technical Documentation](graph_mode_multi_shape_memory_reuse.md)

## Usage

These features are implemented inside the xLLM engine and are transparent to users. Enable Graph Mode with the gflags parameter `enable_graph` (default: false). Set it to true in the xLLM service startup script, for example:

```shell
--enable_graph=true
```

## Performance Impact

- With Graph Mode enabled, decode-phase throughput **improves by about 8%–10%** on models such as Qwen3-0.6B and Qwen3-1.7B.

## Model Support

The following table lists each model’s support on ACLGraph, CudaGraph, and MLUGraph.

| Model | ACLGraph | CudaGraph | MLUGraph |
|------|----------|-----------|----------|
| Qwen3/Qwen3-MoE | ✅ | ✅ | ✅ |
| DeepseekV3.2 | ✅ | | |
| GLM4.5/4.6/4.7 | ✅ | | |
| Qwen2.5-VL | | | ✅ |

!!! warning "Adding Graph Mode support for new models"
    Ensure that the kernels used in the computation implement dynamic dimension parameterization; otherwise the graph may break and kernels may need to be re-implemented.

## Related Documentation

- [Graph Mode Multi-Shape Memory Reuse Technical Documentation](graph_mode_multi_shape_memory_reuse.md)
