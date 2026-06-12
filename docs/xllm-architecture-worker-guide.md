# xLLM 框架架构与 Worker 执行流程详解

## 1. 概览

xLLM 是一个高性能的 LLM 推理框架，支持多种硬件平台（NPU、CUDA、MLU、DCU），提供分布式推理、PD disaggregation、推测解码等高级特性。

### 1.1 架构层级图

```
┌─────────────────────────────────────────────────────────────┐
│                      HTTP API Service                        │
│                   (opencode_server/c_api)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                        Master                                │
│  ┌──────────────┬──────────────┬──────────────┬───────────┐│
│  │  LLMMaster   │  VLMMaster   │  RecMaster   │ DiTMaster ││
│  └──────────────┴──────────────┴──────────────┴───────────┘│
│                                                              │
│  • Scheduler (ContinuousScheduler/FixedStepsScheduler)      │
│  • Request Queue Management                                  │
│  • Tokenizer & ChatTemplate                                  │
│                                                              │
│  engine_: std::unique_ptr<Engine>                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                        Engine                                │
│  ┌──────────┬──────────┬──────────┬────────────┬──────────┐│
│  │LLMEngine │VLMEngine │RecEngine │DiTEngine   │SSMEngine ││
│  └──────────┴──────────┴──────────┴────────────┴──────────┘│
│                                                              │
│  • WorkerClient[] (RPC communication)                        │
│  • BlockManagerPool (KV Cache block allocation)              │
│  • Distributed coordination (DistManager)                    │
│                                                              │
│  worker_clients_: vector<WorkerClient>                       │
└─────────────────────┬───────────────────────────────────────┘
                      │ (RPC over brpc/gRPC)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    WorkerServer                              │
│  • RPC server endpoint                                       │
│  • Receives step() requests from Engine                      │
│  • Routes to Worker instance                                 │
│                                                              │
│  worker_: std::unique_ptr<Worker>                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                        Worker                                │
│  ┌──────────────┬──────────────┬──────────────┬───────────┐│
│  │LLMWorkerImpl │VLMWorkerImpl │RecWorkerImpl │EmbedWorker││
│  └──────────────┴──────────────┴──────────────┴───────────┘│
│                                                              │
│  • Model loading and initialization                          │
│  • KV Cache management                                       │
│  • Sampling (Sampler/BeamSearcher)                           │
│  • Forward execution coordination                            │
│                                                              │
│  model_: CausalLM*                                           │
│  model_executor_: Executor                                   │
│  kv_caches_: vector<KVCache>                                 │
│  sampler_: Sampler                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                       Executor                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                ExecutorImpl                              ││
│  ├────────────┬────────────┬────────────┬─────────────────┤│
│  │BaseExecutor│CudaGraph   │AclGraph    │VlmExecutor      ││
│  │Impl        │ExecutorImpl│ExecutorImpl│Impl             ││
│  └────────────┴────────────┴────────────┴─────────────────┘│
│                                                              │
│  • Platform-specific optimization (Graph/Eager)              │
│  • Calls model_->forward()                                   │
│                                                              │
│  model_: CausalLM*                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     Model (CausalLM)                         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              LlmForCausalLMImplBase                      ││
│  ├──────────────┬──────────────┬──────────────┬───────────┤│
│  │LlamaForCausal│Qwen2ForCausal│DeepSeekFor   │OneRec     ││
│  │LM            │LM            │CausalLM      │           ││
│  └──────────────┴──────────────┴──────────────┴───────────┤│
│                                                              │
│  • Transformer forward (embedding → layers → norm)           │
│  • LM head (logits computation)                              │
│                                                              │
│  model_: LlmModelImplBase (Transformer)                      │
│  lm_head_: LmHead                                            │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 完整调用链

```
HTTP Request
  → LLMMaster::handle_request()
    → RequestQueue.enqueue()
  
LLMMaster::run() [scheduler loop]
  → ContinuousScheduler::schedule()
    → Batch creation
    → LLMEngine::step()
      → worker_clients_[i]->step() [RPC]
        → WorkerServer::step()
          → Worker::step()
            → LLMWorkerImpl::step_internal()
              → model_executor_->forward()
                → ExecutorImpl::run()
                  → CausalLM::forward()
                    → LlmModelImplBase::forward()
                      → embed_tokens_ → layers_ → norm_
              → sampler_->forward()
              → return ForwardOutput
        → Return to Engine
      → Engine aggregates results
    → Scheduler processes outputs
      → OutputCallback (send to client)
```

---

## 2. Worker 详细分析

### 2.1 Worker 的核心职责

Worker 是 xLLM 中实际执行模型推理的核心组件，负责：

1. **模型加载与初始化**
   - 从 checkpoint 加载模型权重
   - 创建 CausalLM 实例
   - 初始化 Executor

2. **KV Cache 管理**
   - 分配和管理 KV Cache blocks
   - 处理 cache 的分配、释放、copy

3. **推理执行协调**
   - 调用 Executor 执行 forward pass
   - 处理 sampling（sampling params, beam search）
   - 返回 logits 和采样结果

4. **平台适配**
   - 设备管理（CUDA/NPU/MLU/DCU）
   - Stream 同步和事件管理

### 2.2 Worker 的类结构

#### 2.2.1 接口层（Pimpl 模式）

```cpp
// core/runtime/worker.h
class Worker {
 public:
  Worker(const runtime::Options& options, ...);
  
  std::optional<ForwardOutput> step(const ForwardInput& input);
  std::optional<ForwardOutput> step_no_sync(const ForwardInput& input);
  folly::SemiFuture<std::optional<ForwardOutput>> step_async(const ForwardInput& input);
  
 private:
  std::unique_ptr<WorkerImpl> impl_;  // Pimpl
};
```

#### 2.2.2 实现层继承结构

```
WorkerImpl (基类)
  ├─ LLMWorkerImpl      - 标准 LLM 推理
  ├─ VLMWorkerImpl      - Vision-Language Model
  ├─ RecWorkerImpl      - 推荐模型（OneRec/LlmRec）
  ├─ EmbedWorkerImpl    - Embedding 模型
  └─ DiTWorkerImpl      - Diffusion Transformer
```

#### 2.2.3 LLMWorkerImpl 核心成员

```cpp
// core/runtime/llm_worker_impl.h
class LLMWorkerImpl : public WorkerImpl {
 private:
  // 核心成员
  std::unique_ptr<CausalLM> model_;           // 模型实例
  std::unique_ptr<Executor> model_executor_;  // 执行器
  std::vector<KVCache> kv_caches_;            // KV Cache
  std::unique_ptr<Sampler> sampler_;          // 采样器
  std::unique_ptr<BeamSearcher> beam_searcher_; // Beam Search
  
  // 平台相关
  Device device_;                             // 设备（cuda/npu/mlu）
  ModelContext context_;                      // 模型上下文
  
  // 可选组件
  std::unique_ptr<EplbExecutor> eplb_executor_;  // Expert parallel load balancing
  std::unique_ptr<Driver> driver_;               // Speculative decode driver
};
```

### 2.3 Worker 执行一个请求的完整流程

#### Step 1: 初始化阶段

```cpp
// llm_worker_impl.cpp:80-98
bool LLMWorkerImpl::init_model(ModelContext& context) {
  // 1. 创建模型实例
  model_ = create_llm_model(context);  // 通过工厂模式创建
  
  // 2. 创建 Executor
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);
  
  // 3. 初始化可选组件
  if (EPLBConfig::enable_eplb()) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }
  
  if (BeamSearchConfig::enable_beam_search_kernel()) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }
  
  return true;
}
```

#### Step 2: 接收请求（ForwardInput）

```cpp
// ForwardInput 结构（从 Engine 传递）
struct ForwardInput {
  torch::Tensor token_ids;        // [num_tokens]
  torch::Tensor positions;        // [num_tokens]
  
  ForwardInputParams input_params; // 包含:
    - SamplingParams (temperature, top_p, top_k, etc.)
    - AttentionParams (seq_lens, kv_seq_lens)
    - ExpertParams (for MoE)
    - BlockCopyParams (KV cache copy)
    
  std::vector<TransferKVInfo> transfer_kv_infos;  // P-D disaggregation
};
```

#### Step 3: 执行前准备

```cpp
// llm_worker_impl.cpp:101-107
std::optional<ForwardOutput> LLMWorkerImpl::step_no_sync(const ForwardInput& input) {
  // 1. 数据准备（copy to device, prepare input tensors）
  ForwardInput input_on_device;
  prepare_work_before_execute(input, input_on_device);
  
  // 2. 获取当前 stream
  std::unique_ptr<Stream> current_stream = device_.current_stream();
  
  // 3. 执行
  return execute_no_sync_on_stream(input_on_device, *current_stream);
}
```

#### Step 4: 核心 Forward 执行

```cpp
// llm_worker_impl.cpp:196-241
std::optional<ForwardOutput> LLMWorkerImpl::step_internal(
    const ForwardInput& input, ForwardSyncPolicy sync_policy) {
  
  // 1. 可选: KV Cache transfer (P-D disaggregation)
  if (!input.transfer_kv_infos.empty()) {
    kv_cache_transfer_(input.transfer_kv_infos, ...);
  }
  
  // 2. 可选: Expert parallel load balancing
  if (eplb_executor_) {
    eplb_executor_->eplb_execute(input.input_params.expert.eplb_info);
  }
  
  // 3. ★ 核心: 调用 Executor forward
  auto model_output = model_executor_->forward(
      input.token_ids, 
      input.positions, 
      kv_caches_, 
      input.input_params);
  
  // 4. 检查是否需要继续
  if (!model_output.hidden_states.defined()) {
    return std::nullopt;  // 无输出（如纯 prefill）
  }
  
  // 5. 计算 logits（选择性计算）
  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = model_->logits(model_output.hidden_states, selected_token_idxes);
  }
  
  // ...
}
```

#### Step 5: Sampling 和结果返回

```cpp
// llm_worker_impl.cpp:303-327
  // 6. Sampling（生成 token）
  if (sampling_params.selected_token_idxes.defined()) {
    output.logits = logits;
    
    if (!input.skip_sampling_for_logits_only) {
      // 调用 sampler 生成下一个 token
      auto sample_output = sampler_->forward(logits, sampling_params);
      
      // Beam Search（如果启用）
      BeamSearchOutput beam_search_output;
      if (sampling_params.use_beam_search) {
        beam_search_output = beam_searcher_->forward(...);
      }
      
      output.sample_output = sample_output;
      output.beam_search_output = beam_search_output;
    }
  }
  
  return output;
}
```

#### Step 6: 返回结果

```cpp
// ForwardOutput 结构
struct ForwardOutput {
  torch::Tensor logits;              // [num_seqs, vocab_size]
  
  SampleOutput sample_output;        // 包含:
    - torch::Tensor sampled_tokens   // 采样的 token ids
    - torch::Tensor top_tokens       // top-k tokens
    - torch::Tensor top_logprobs     // 对应 logprobs
    
  BeamSearchOutput beam_search_output;  // Beam Search 结果
  
  // 其他元数据
  torch::Tensor expert_load_data;    // EPLB 相关
  int32_t prepared_layer_id;         // Rolling load 相关
};
```

### 2.4 Worker 类图详解

```
┌──────────────────────────────────────────────────────────────┐
│                      Worker (接口)                            │
│  • step() / step_no_sync() / step_async()                    │
│  • init_model() / init_kv_cache()                            │
└───────────────────┬──────────────────────────────────────────┘
                    │ [Pimpl]
                    ▼
┌──────────────────────────────────────────────────────────────┐
│                   WorkerImpl (基类)                           │
│  • init_model() - 创建 model_ 和 executor_                   │
│  • init_kv_cache() - 分配 KV Cache blocks                    │
│  • step_internal() - 执行 forward + sampling                 │
│                                                              │
│  model_: std::unique_ptr<CausalLM>                           │
│  model_executor_: std::unique_ptr<Executor>                  │
│  kv_caches_: std::vector<KVCache>                            │
│  sampler_: std::unique_ptr<Sampler>                          │
└───────────────────┬──────────────────────────────────────────┘
                    │ [继承]
        ┌───────────┴───────────┬──────────────────┐
        ▼                       ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│LLMWorkerImpl │    │VLMWorkerImpl │    │RecWorkerImpl │
│              │    │              │    │              │
│• 标准 LLM    │    │• VLM 特化    │    │• 推荐模型    │
│• sampling    │    │• image_proc  │    │• OneRec/LlmRec│
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 3. Executor 规格详解

### 3.1 Executor 的设计目的

Executor 是 Worker 与 Model 之间的执行桥梁，主要负责：

1. **平台特化优化**
   - Eager mode（BaseExecutorImpl）
   - CUDA Graph optimization（CudaGraphExecutorImpl）
   - ACL Graph optimization（AclGraphExecutorImpl）
   - MLU/DCU Graph optimization

2. **输入准备**
   - 从 Batch 准备 ForwardInput
   - 处理 tensor layout 和 device placement

3. **Forward 调用**
   - 调用 CausalLM::forward()
   - 处理 stream 同步

### 3.2 Executor 类结构

#### 3.2.1 接口定义

```cpp
// core/runtime/executor_impl.h
class ExecutorImpl {
 public:
  virtual ~ExecutorImpl() = default;
  
  // 从 Batch 准备输入
  virtual ForwardInput prepare_inputs(Batch& batch) = 0;
  
  // 执行模型 forward
  virtual ModelOutput run(
      const torch::Tensor& tokens,
      const torch::Tensor& positions,
      std::vector<KVCache>& kv_caches,
      const ModelInputParams& params) = 0;
};
```

#### 3.2.2 封装层

```cpp
// core/runtime/executor.h
class Executor {
 public:
  Executor(CausalLM* model, const ModelArgs& args, ...);
  
  ForwardInput prepare_inputs(Batch& batch) {
    return impl_->prepare_inputs(batch);
  }
  
  ModelOutput forward(const torch::Tensor& tokens, ...) {
    return impl_->run(tokens, positions, kv_caches, params);
  }
  
 private:
  std::unique_ptr<ExecutorImpl> impl_;  // Pimpl
};
```

#### 3.2.3 工厂创建逻辑

```cpp
// core/runtime/executor.cpp:24-33
Executor::Executor(CausalLM* model, ...) {
  // 根据 backend 和 enable_graph 选择 ExecutorImpl
  std::string backend = (options.backend() != "vlm" && options.enable_graph())
                             ? Device::type_str()
                             : options.backend();
  
  impl_ = ExecutorImplFactory::get_instance().create_executor_impl(
      model, args, device, options, backend);
}
```

### 3.3 ExecutorImpl 实现规格

```
ExecutorImpl (接口)
  │
  ├─ BaseExecutorImpl
  │   • 最基础的 Eager 实现
  │   • 直接调用 model_->forward()
  │   • 无特殊优化
  │
  ├─ CudaGraphExecutorImpl
  │   • CUDA Graph 优化
  │   • 预编译 graph，减少 kernel launch overhead
  │   • decode mode 优化（固定 batch size）
  │   • 支持 piecewise graph（prefill 分段）
  │
  ├─ AclGraphExecutorImpl
  │   • NPU ACL Graph 优化（华为 Ascend）
  │   • ATB（Ascend Tensor Builder）集成
  │   • Rolling weight load 支持
  │   • 支持 speculative decode
  │
  ├─ MluGraphExecutorImpl
  │   • MLU Graph 优化（寒武纪）
  │   • CNGRAPH integration
  │
  ├─ DcuGraphExecutorImpl
  │   • DCU Graph 优化（海光）
  │   • HIP graph integration
  │
  └─ VlmExecutorImpl
      • VLM 特化实现
      • 处理 image embeddings
      • 支持 multimodal forward
```

### 3.4 BaseExecutorImpl 核心实现

```cpp
// core/runtime/base_executor_impl.h
class BaseExecutorImpl : public ExecutorImpl {
 public:
  BaseExecutorImpl(CausalLM* model, const ModelArgs& args, 
                   const torch::Device& device, const runtime::Options& options)
      : model_(model), args_(args), device_(device), options_(options) {}
  
  // 准备输入
  ForwardInput prepare_inputs(Batch& batch) override {
    return batch.prepare_forward_input(
        options_.num_decoding_tokens(), 0, args_, options_.cp_size());
  }
  
  // ★ 核心 run 方法
  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override {
    COUNTER_INC(num_model_execution_total_eager);
    // 直接调用 model forward
    return model_->forward(tokens, positions, kv_caches, params);
  }
  
 private:
  CausalLM* model_;              // 模型指针（不持有）
  ModelArgs args_;               // 模型配置
  torch::Device device_;         // 设备
  runtime::Options options_;     // 运行时选项
};
```

### 3.5 ExecutorImpl 工厂注册

```cpp
// core/runtime/executor_impl_factory.h
class ExecutorImplFactory {
 public:
  // 注册工厂函数
  void register_executor_impl(
      const std::string& backend,
      std::function<std::unique_ptr<ExecutorImpl>(...)> factory) {
    factories_[backend] = factory;
  }
  
  // 创建 ExecutorImpl
  std::unique_ptr<ExecutorImpl> create_executor_impl(
      CausalLM* model, ..., const std::string& backend) {
    auto it = factories_.find(backend);
    if (it != factories_.end()) {
      return it->second(model, args, device, options);
    }
    // 默认返回 BaseExecutorImpl
    return std::make_unique<BaseExecutorImpl>(model, args, device, options);
  }
  
 private:
  std::unordered_map<std::string, FactoryFunc> factories_;
};

// 注册示例（自动注册）
REGISTER_EXECUTOR_IMPL("cuda_graph", [](...) {
  return std::make_unique<CudaGraphExecutorImpl>(model, args, device, options);
});

REGISTER_EXECUTOR_IMPL("acl_graph", [](...) {
  return std::make_unique<AclGraphExecutorImpl>(model, args, device, options);
});
```

---

## 4. Model (CausalLM) 层次结构

### 4.1 CausalLM 接口

```cpp
// core/framework/model/causal_lm.h
class CausalLM : public torch::nn::Module {
 public:
  // ★ 核心 forward 方法
  virtual ModelOutput forward(
      const torch::Tensor& tokens,
      const torch::Tensor& positions,
      std::vector<KVCache>& kv_caches,
      const ModelInputParams& parameters) = 0;
  
  // 计算 logits
  virtual torch::Tensor logits(
      const torch::Tensor& hidden_states,
      const torch::Tensor& seleted_idxes) = 0;
  
  // 加载模型权重
  virtual void load_model(std::unique_ptr<ModelLoader> loader) = 0;
  
  // Pooler（推荐模型）
  virtual torch::Tensor pooler(
      const torch::Tensor& hidden_states,
      const torch::Tensor& seleted_idxes);
};
```

### 4.2 模型实现层次

```
CausalLM (接口)
  │
  └─ CausalLMImpl<Model> (模板包装器)
      │
      ├─ LlmForCausalLMImplBase<LlmModelType> (NPU/CUDA common base)
      │   │
      │   ├─ LlamaForCausalLM
      │   │   └─ model_: LlamaModel
      │   │       ├─ embed_tokens_
      │   │       ├─ layers_[] (DecoderLayer)
      │   │       └─ norm_
      │   │   └─ lm_head_
      │   │
      │   ├─ Qwen2ForCausalLM
      │   ├─ DeepSeekForCausalLM
      │   ├─ Qwen3ForCausalLM
      │   └─ ...
      │
      ├─ RecModelImplBase (推荐模型)
      │   ├─ OneRecForCausalLM
      │   └─ ...
      │
      └─ CausalVLM (VLM)
          └─ Qwen2VLForCausalLM
          └─ Qwen3VLForCausalLM
```

### 4.3 模型注册机制

```cpp
// models/llm/npu/llama.h:313
REGISTER_CAUSAL_MODEL(llama, LlamaForCausalLM);

// models/model_registry.cpp:334
std::unique_ptr<CausalLM> create_llm_model(const ModelContext& context) {
  std::string resolved_name = resolve_model_registration_name(context.model_type());
  
  auto factory = ModelRegistry::get_causallm_factory(resolved_name);
  if (factory) {
    return factory(context);  // 调用注册的工厂函数
  }
  
  return nullptr;
}
```

---

## 5. KV Cache 管理

### 5.1 KV Cache 结构

```cpp
// core/framework/kv_cache/kv_cache.h
class KVCache {
 public:
  torch::Tensor get_k_cache();  // [num_blocks, block_size, num_heads, head_dim]
  torch::Tensor get_v_cache();
  
  int32_t device_index();
  int32_t layer_id();
  
 private:
  torch::Tensor k_cache_;
  torch::Tensor v_cache_;
  BlockManager* block_manager_;  // 块分配管理器
};
```

### 5.2 Block Manager

```cpp
// core/framework/block/block_manager.h
class BlockManager {
 public:
  // 分配 blocks
  std::vector<int64_t> allocate(int64_t num_blocks);
  
  // 释放 blocks
  void free(const std::vector<int64_t>& block_ids);
  
  // Block copying（for prefix caching）
  void copy_blocks(const std::vector<int64_t>& src_blocks,
                   const std::vector<int64_t>& dst_blocks);
  
 private:
  std::vector<Block> blocks_;
  std::vector<int64_t> free_blocks_;
  int64_t num_total_blocks_;
  int64_t num_free_blocks_;
};
```

---

## 6. 关键设计模式总结

### 6.1 Pimpl 模式（Pointer to Implementation）

用于 Worker 和 Executor，隔离接口与实现：
- `Worker` → `WorkerImpl`
- `Executor` → `ExecutorImpl`

优点：
- 隐藏实现细节
- 减少 header 依赖
- ABI 稳定性

### 6.2 工厂模式

用于模型和 Executor 创建：
- `create_llm_model()` + `REGISTER_CAUSAL_MODEL`
- `ExecutorImplFactory` + `REGISTER_EXECUTOR_IMPL`

优点：
- 可扩展性强
- 配置驱动创建

### 6.3 CRTP（Curiously Recurring Template Pattern）

用于模型基类：
- `LlmForCausalLMImplBase<LlmModelType>`
- `LlmModelImplBase<DecoderLayerType>`

优点：
- 静态多态
- 避免虚函数开销

### 6.4 策略模式

用于 Scheduler：
- `ContinuousScheduler` - 流式推理
- `FixedStepsScheduler` - 固定步数（推荐模型）

---

## 7. 完整执行示例（LLM）

### 7.1 用户请求流程

```
1. HTTP Request: POST /v1/chat/completions
   Body: {"messages": [...], "temperature": 0.7}

2. LLMMaster::handle_request()
   - Tokenize prompt
   - Create Request object
   - Enqueue to RequestQueue

3. LLMMaster::run() [scheduler thread]
   - ContinuousScheduler::schedule()
   - Select requests from queue
   - Create Batch (group multiple requests)

4. LLMEngine::step(batch)
   - Prepare ForwardInput for each worker
   - Call worker_clients_[i]->step() [RPC]

5. WorkerServer::step() [on worker process]
   - Receive RPC request
   - Call Worker::step()

6. LLMWorkerImpl::step_internal()
   - model_executor_->forward()
     → ExecutorImpl::run()
       → CausalLM::forward()
         → Transformer forward (embedding → layers → norm)
   - sampler_->forward(logits) [decode mode]
   - Return ForwardOutput

7. Engine aggregates results
   - Collect outputs from all workers
   - Return to Scheduler

8. Scheduler processes outputs
   - Update request states
   - Decode sampled tokens
   - Call OutputCallback [stream to client]

9. Response: {"choices": [...], "usage": {...}}
```

### 7.2 Prefill vs Decode

**Prefill Phase:**
```cpp
// 输入: prompt tokens [prompt_len]
// 输出: hidden_states only (no sampling)
ForwardOutput {
  logits = undefined,
  sample_output = undefined,
  hidden_states = [prompt_len, hidden_size]
}
```

**Decode Phase:**
```cpp
// 输入: sampled token [1] + KV cache
// 输出: logits + sampled token
ForwardOutput {
  logits = [1, vocab_size],
  sample_output = {
    sampled_tokens = [1],
    top_logprobs = [...]
  }
}
```

---

## 8. 性能优化特性

### 8.1 CUDA/ACL Graph

- 预编译 kernel sequence
- 减少 CPU overhead
- 固定 decode batch size 优化

### 8.2 Continuous Batching

- 动态 batch 组合
-最大化 GPU 利用率
- Iteration-level scheduling

### 8.3 Prefix Caching

- Cache common prompt prefixes
- Block reuse across requests
- KV cache copy optimization

### 8.4 P-D Disaggregation

- Prefill 和 Decode 分离
- KV cache transfer over network
- Independent scaling

### 8.5 Speculative Decoding

- Draft model 提前生成
- Target model verify
- 提升吞吐

---

## 9. 总结

xLLM 的架构设计体现了高度的模块化和可扩展性：

1. **清晰的层次分离**：Master → Engine → Worker → Executor → Model
2. **灵活的平台适配**：通过 ExecutorImpl 工厂支持多硬件
3. **高效的通信机制**：RPC 分布式 + 异步 callback
4. **丰富的优化策略**：Graph、Continuous Batching、Prefix Cache、Speculative Decode

Worker 作为核心执行单元，通过组合 Model、Executor、Sampler 等组件，实现了完整的推理流程，同时保持了良好的扩展性和性能优化能力。