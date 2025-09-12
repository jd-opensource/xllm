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