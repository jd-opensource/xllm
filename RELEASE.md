# Release xllm 0.7.0

## **Major Features and Improvements**

### Model Support

- Support GLM-4.5.
- Support Qwen3-Embedding.
- Support Qwen3-VL.
- Support FluxFill.

### Feature
- Support MLU backend, currently supports Qwen3 series models.
- Support dynamic disaggregated PD, with dynamic switching between P and D phases based on strategy.
- Support multi-stream parallel overlap optimization.
- Support beam-search capability in generative models.
- Support virtual memory continuous kv-cache capability.
- Support ACL graph executor.
- Support unified online-offline co-location scheduling in disaggregated PD scenarios.
- Support PrefillOnly Scheduler.
- Support v1/rerank model service interface.
- Support communication between devices via shared memory instead of RPC on a single machine.
- Support function call.
- Support reasoning output in chat interface.
- Support top-k+add fusion in the router component of MoE models.
- Support offline inference for LLM, VLM, and Embedding models.
- Optimized certain runtime performance.

### Bugfix
- Skip cancelled requests when processing stream output.
- Resolve segmentation fault during qwen3 quantized inference.
- Fix the alignment of monitoring metrics format for Prometheus.
- Clear outdated tensors to save memory when loading model weights.
- Fix attention mask to support long sequence requests.
- Fix bugs caused by enabling scheduler overlap.

# Release xllm 0.6.0

## **Major Features and Improvements**

### Model Support

- Support DeepSeek-V3/R1.
- Support DeepSeek-R1-Distill-Qwen.
- Support Kimi-k2.
- Support Llama2/3.
- Support Qwen2/2.5/QwQ.
- Support Qwen3/Qwen3-MoE.
- Support MiniCPM-V.
- Support MiMo-VL.
- Support Qwen2.5-VL .

### Feature

- Support KV cache store.
- Support Expert Parallelism Load Balance.
- Support multi-priority on/offline scheduler.
- Support latency-aware scheduler.
- Support serving early stop.
- Optimize ppmatmul kernel.
- Support image url input for VLM.
- Support disaggregated prefill and decoding.
- Support large-scale EP parallelism.
- Support Hash-based PrefixCache matching.
- Support Multi-Token Prediction for DeepSeek.
- Support asynchronous scheduling, allowing the scheduling and computational pipeline to execute in parallel.
- Support EP, DP, TP model parallel.
- Support multiple process and multiple nodes.

### Docs

- Add getting started docs.
- Add features docs.