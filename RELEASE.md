# Release xllm 0.8.0

## **Major Features and Improvements**

### Model Support

#### NPU
- Support DeepSeek-v3.2 model.
- Support GLM4.7 model.
- Support GLM4.6Vmodel.
- Support GME-Qwen2-VL model.
- Support FluxControl model.
#### CUDA
- Support Qwen2/3 Dense model.
#### MLU
- Support DeepSeek-v3.2 model.
- Support Qwen2_5_vl/Qwen3_vl/Qwen3_vl_moe model.
#### ILU
- Support Qwen3-0.6B model.

### Feature
- Implement chunked prefill and prefix cache for Qwen3 MoE.
- Support GLM-4.6V model.
- Add wrappers for ATB and ACLNN fused operators.
- Optimize prefetch from kv cache store.
- Support Qwen2-VL & GME-Qwen2-VL model on npu device.
- Fix hang issue when enable schedule overlap.
- Add GLM-4.7 detector implementation and update tool call parser.
- Adapt hierarchy block manager for disagg PD.
- Support deepseek-v3.2-Exp for npu.
- Support acl_graph for qwen3/qwen3_moe.
- Support prefix cache for deepseek-v3/r1 models.
- Support disagg PD for MTP.
- Add mooncake kv cache transfer.
- Add GLM-4.7 support to reasoning detector registry.
- Support nd-to-nz continuous memory copy.
- Support RPC-based link/unlink for PD disaggregation.
- Support IntraLayerAddNorm, aclgraph, etc for DeepSeek V3.2.
- Add activation, norm and rope ops for cuda device.
- Support fused norm for Qwen3 and DeepSeek for cuda device.
- Build deepseek v2 decoder layer and related model files for mlu device.
- Support qwen2_5_vl/qwen3_vl/qwen3_vl_moe on mlu device.
- Add moe all2all kernels and deep ep layer on mlu device.
- Support deepseek mtp on mlu device.
- Support graph executor on mlu device.
- Support dp+ep moe and all2all computation on mlu device.
- Support parallelized shared experts in fused moe on mlu device.
- Support qwen3 0.6B model on iluvatar device.
- Add rec proto,serivce and utils for rec framework
- Support C api for llm inference.
- Add constrained decoding for generative recommendation.
- Add rec scheduler master and engine for rec framework.
- Add rec_type and onerec batch input builder for rec framework.
- Add onerec worker impl for rec framework.
- Add qwen3/LlmRec support in rec framework.

### Bugfix
- Reslove core dump of stream chat completion request when backend is VLM.
- Resolve duplicate content in multi-turn tool call conversations.
- Fix core dump issue triggered by client disconnection.
- Fix the memory leak issue in the completions interface.
- Fix wrong positons of validate input when enable MTP.
- Resolve kv_cache_num mismatch in ChunkedPrefill due to H2D block copy.
- Fix the missing index shape in the allocate kv cache transfer.
- Fix MiMo-VL weights loading crash on NPU device.
- Fix inaccurate metrics issue when enabling schedule overlap.
- Fix potential out-of-range and block leaks during deallocate in D2H copy.
- Fix allocation failure in HierarchyBlockManagerPool::allocate.
- Fix deepseek accuracy issues with prefix cache enabled.
- Resolve Deepseek execution failure caused by invalid input.
- Fix DeepSeek failing to run when enabling DP.
- Fix the rate_limit bug for stream and non-stream request in PD disagg and refactor some callback logics.
- Correct attn mask when prefix cache and MTP are both enabled in deepseek.
- Correct precision loss when enabling prefixcache with disagg pd.
- Fix incorrect async implementation in rerank interface.
- Fix acl_graph_executor not handling q_cu_seq_lens parameter for deepseekv3.2.
- Fix precision issue when enabling MTP in PD disaggregation mode.
- Fix mrope calculation in the multimodal situation.
- Fix core dump of large beam width.

# Release xllm 0.7.2

## **Major Features and Improvements**

### Feature

- Enhance Qwen3-MoE to support TP settings beyond 4.
- Implement chunked prefill and prefix cache for Qwen3 MoE.
- Support prefix cache for DeepSeek-V3/R1 models.

### Bugfix

- Fix core dump issue triggered by client disconnection. 
- Fix the incorrect reading of model args from Qwen3-VL's config.json.
- Setup the tokenizer config function of bos and eos to fast tokenizer.
- Fix the memory leak issue.
- Fix hang issue when enable schedule overlap.

# Release xllm 0.7.1

## **Major Features and Improvements**

### Model Support

- Support GLM-4.5-Air.
- Support Qwen3-VL-Moe.

### Feature

- Support scheduler overlap when enable chunked prefill and MTP.
- Enable multi-process mode when running VLM model.
- Support AclGraph for GLM-4.5.

### Bugfix

- Reslove core dump of qwen embedding 0.6B.
- Resolve duplicate content in multi-turn tool call conversations.
- Support sampler parameters for MTP.
- Enable MTP and schedule overlap to work simultaneously.
- Resolve google.protobuf.Struct parsing failures which broke tool_call and think toggle functionality.
- Fix the precision issue in the Qwen2 model caused by model_type is not be assigned.
- Fix core dump of GLM 4.5 when enable MTP.
- Temporarily use heap allocation for VLM backend.
- Reslove core dump of stream chat completion request for VLM.

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
