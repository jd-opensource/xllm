# ACLGraph

## Feature Description

To optimize Host-side scheduling performance, NPU recently introduced ACLGraph, a graph mode solution similar to CUDA Graph. Compared to the traditional mode that uses CPU-intensive small task submission and frequent small Kernel launches on NPU, ACLGraph mode significantly reduces startup time and NPU bubbles by submitting large tasks from the CPU once and then executing small kernels in a streaming manner within the NPU.

By implementing the ACLGraph functionality in the xLLM engine, we have achieved the following features:

### Dynamic Shape Parameterization
  - Key dynamic shapes (such as batch size and sequence length) are treated as whole-graph input parameters, thereby enhancing flexibility. During memory allocation and kernel configuration for the graph, these dynamic parameters are used to calculate actual required values, for example, calculating the block_table size using the formula $block\_table\_size = batch\_size \times (max\_seq\_len / block\_size)$. During the graph launch phase, the actual batch size and maximum sequence length are passed as parameters to ensure that kernels can use the correct strides to access data.

### Multi-Shape Reusable Memory Pool
  - To avoid waste caused by using separate memory buffers (for input, output, and intermediate Tensors) for different shapes, we employ an expandable memory pool. Multiple shapes reuse the base address of the pool, with different shapes having different offsets from the pool's base address.

## Usage

The aforementioned features have been implemented internally within the xLLM engine and are transparent to users. Users do not need to concern themselves with the internal implementation details and can directly enable the relevant functionality in applicable scenarios. Enable it via the gflags parameter `enable_graph`. The parameter defaults to false. To enable it, set it to true in the xLLM service startup script, as shown in the example below:
```shell
--enable_graph=true
```

## Performance Impact
- After enabling the ACLGraph function, the decode phase throughput **improves by 8%-10%** on models such as Qwen3-0.6B and Qwen3-1.7B.

!!! warning "Important Notes"
    - When adding ACLGraph support for a new model, it is necessary to check whether the kernels used in the computation process have implemented dynamic dimension parameterization. If not, the kernels need to be re-implemented.

!!! tip "Future Plans"
    * Support communication operations between Attention DP and FFN EP in MoE models to adapt to different shapes.