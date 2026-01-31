# Overall Architecture

## Backgroud

In recent years, with the groundbreaking progress of large language models (LLMs) ranging from tens of billions to trillions of parameters (such as GPT, Claude, DeepSeek, LLaMA, etc.) in the fields of natural language processing and multimodal interaction, the industry has an urgent need for efficient inference engines and service systems. How to reduce cluster inference costs and improve inference efficiency has become a key challenge for achieving large-scale commercial deployment.

Although a number of optimization engines for large model inference have emerged, several technical bottlenecks remain in practical deployment:

*   **Hardware Adaptability Challenges:** Existing inference engines lack sufficient support for the architectural characteristics of specialized accelerators like domestic chips, making it difficult to fully exploit the performance potential of heterogeneous computing hardware, leading to low computational resource utilization.
*   **MoE Architecture Optimization Difficulties:** The token distribution process in expert parallelism mechanisms generates significant All-to-All communication overhead, while dynamic routing strategies cause expert load imbalance, severely constraining system scalability.
*   **Long Context Management Bottlenecks:** As model context windows continue to expand, the efficiency of optimizations in KV cache handling—such as memory fragmentation management and cross-node synchronization—directly impacts overall inference throughput performance.
*   **Hybrid Deployment Efficiency Limitations:** Existing inference clusters struggle to simultaneously guarantee service quality (SLO) and optimize resource utilization when handling both online services and offline tasks.
*   **Insufficient Dynamic PD Adaptation:** When input/output sequence lengths fluctuate drastically, static PD resource partitioning lacks the ability to adjust PD resource configurations in real-time. This can lead to idle GPU resources and poses a risk of SLO violations.

To address these challenges, we present xLLM—an efficient and user-friendly open-source intelligent inference framework that provides enterprise-grade service guarantees and high-performance engine computing capabilities for model inference on domestic chips.

## Feature Introduction

xLLM provides intelligent computing capabilities, implementing joint inference acceleration across multiple computational system layers and algorithm-driven layers:

### Computational System Layer

#### Multi-Level Pipeline Execution Orchestration
Asynchronizes CPU scheduling at the framework layer to pipeline it with chip inference computation, reducing computation bubbles; at the model graph layer, splits single batches to create pipelines between micro-batches, overlapping computation and communication; at the operator kernel layer, pipelines different computing units, overlapping computation and memory access.
#### Dynamic Shape Graph Execution Optimization
Addressing the static graph adaptation problem for large language models processing dynamic inputs (e.g., variable sequence lengths and batch sizes), xLLM achieves dynamic adaptation through parametric design capturing input dimensions. It combines a multi-graph caching scheme to reduce compilation overhead and uses a managed memory pool instead of absolute addresses to ensure safe reuse, ultimately achieving high execution efficiency while maintaining high flexibility.
#### Operator Optimization
xLLM implements specific optimizations for key operators in LLMs on domestic hardware chips, including GroupMatmul, Chunked Prefill, and others.
#### xTensor Memory Management
The xTensor memory management framework employs a method of *pre-allocated physical memory page pools + contiguous virtual address mapping*. It achieves efficient dynamic memory management through dynamic on-demand mapping of physical pages, reuse of reusable memory pages (Reusable), and optimized scheduling with asynchronous pre-mapping. Combined with NPU operator adaptation (such as virtual address FlashMLA), it results in improved memory utilization and reduced latency.

### Algorithm-Driven Layer

#### PD Separation
xLLM fully supports PD separation scenarios, enabling efficient management of PD instances, communication between PD instances, and KV cache transfer.
#### Global Scheduling
xLLM provides intelligent, full-lifecycle resource scheduling management for requests and instances.
##### Instance Scheduling
We have implemented various instance scheduling strategies to select how to assign instances to more suitable ones. These include a simple Round Robin strategy, a prefix cache-aware strategy based on the prefix cache hit rate of requests on each instance, and a KV cache-aware strategy based on the free memory level of instances. Furthermore, for PD separation scenarios, where static PD ratios often struggle with traffic fluctuations and sudden changes in request input/output lengths, we implemented an adaptive PD dynamic scheduler responsible for global instance allocation for online requests and runtime PD dynamic adjustment.
##### Request Scheduling
We have implemented various request scheduling strategies supporting continuous batching, including chunked prefill, prefill priority, decode priority, and other batch strategies, all while fully supporting PD separation scenarios.
#### Global KV Cache Management
Utilizes ETCD as a metadata service middleware at the global level for cluster service registration, load information synchronization, and global cache state management. Each compute instance maintains a local multi-level cache pool. Regarding scheduling strategy, the system adopts a dynamic decision-making mechanism based on KV cache: it first performs prefix matching detection, calculates the KV cache reuse rate of candidate nodes, and finally selects the node with the optimal comprehensive performance for processing, achieving dynamic offloading and migration of KV cache.
#### Speculative Inference
xLLM incorporates an optimized speculative inference algorithm that generates multiple tokens at once to boost throughput. xLLM reduces communication costs by下沉 (sinking) the speculative module and optimizes speculative inference computation through methods like overlapping scheduling and computation timelines and reducing operator data movement in speculative scenarios.
#### MoE Load Balancing
xLLM implements expert weight updates based on historical expert load statistics for MoE models. During inference, it achieves effective dynamic load balancing through efficient expert load statistics and double-buffered, seamless expert weight updates.

### Multimodal Support

xLLM provides comprehensive support for various multimodal models, including Qwen2-VL and MiniCPMV.
