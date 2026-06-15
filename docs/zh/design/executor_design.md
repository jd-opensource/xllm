# Executor 执行器设计文档

## 概述

xLLM 的 Executor 执行器是推理引擎的核心组件，负责协调模型的前向计算执行流程。通过统一的抽象接口和平台特化的实现，xLLM 支持多种模型类型（LLM、VLM、DiT、Rec）和多种硬件后端（CUDA、NPU、DCU、MLU）的高效推理。

本文档面向需要理解 Executor 实现原理和架构设计的开发者，重点介绍以下内容：

- Executor 的层次结构与抽象设计
- 各执行器的功能与实现细节
- Graph Mode 优化原理
- 多模态与 MoE 场景的特殊处理
- 执行器选择策略与执行流程

本文档的设计目标包括：

- 统一 xLLM Executor 在不同后端和模型类型上的抽象
- 解释 Graph Executor 的内存池、持久化参数与分桶策略
- 说明 VLM Executor 的编码器缓存机制与 MoE Executor 的异步权重加载

本文档的非目标包括：

- 不覆盖所有算子或所有模型的适配细节
- 不替代功能文档中的参数说明与使用示例

相关设计文档：

- 若希望了解 Graph Mode 的详细设计与动态维度参数化，可参考：[Graph Mode 设计文档](graph_mode_design.md)
- 若希望了解生成式推荐模型的设计，可参考：[生成式推荐设计文档](generative_recommendation_design.md)

## 1. Executor 层次结构

### 1.1 统一抽象接口

xLLM 通过 `ExecutorImpl` 抽象基类定义统一接口，所有具体执行器必须继承并实现以下方法：

```cpp
class ExecutorImpl {
 public:
  virtual ~ExecutorImpl() = default;

  virtual ForwardInput prepare_inputs(Batch& batch) = 0;

  virtual ModelOutput run(const torch::Tensor& tokens,
                          const torch::Tensor& positions,
                          std::vector<KVCache>& kv_caches,
                          const ModelInputParams& params) = 0;
};
```

**核心职责**：

- `prepare_inputs`：准备前向计算的输入数据（tokens、positions、KV Cache 等）
- `run`：执行模型前向计算并返回输出结果（logits、hidden_states）

### 1.2 执行器层次架构

```
┌─────────────────────────────────────────────────────┐
│                   Executor (封装类)                  │
│  - 持有 ExecutorImpl 实例                            │
│  - 对外暴露统一接口                                   │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│               ExecutorImpl (抽象基类)                │
│  - prepare_inputs(Batch& batch)                     │
│  - run(tokens, positions, kv_caches, params)        │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬─────────────┐
        ▼               ▼               ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│BaseExecutor  │ │VlmExecutor   │ │CudaGraph     │ │AclGraph      │
│Impl          │ │Impl          │ │ExecutorImpl  │ │ExecutorImpl  │
│              │ │              │ │              │ │              │
│- LLM/Rec模型 │ │- VLM多模态   │ │- CUDA Graph  │ │- NPU Graph   │
│- Eager执行   │ │- 编码器缓存  │ │- 内存池优化  │ │- ACL优化     │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│DcuGraph      │ │MluGraph      │ │DiTExecutor   │ │EplbExecutor  │
│ExecutorImpl  │ │ExecutorImpl  │ │              │ │              │
│              │ │              │ │- DiT模型     │ │- MoE异步加载 │
│- DCU Graph   │ │- MLU Graph   │ │- 无KV Cache  │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

**设计原则**：

- **统一接口**：所有执行器遵循相同接口，便于上层调用者统一处理
- **平台特化**：不同硬件后端使用专用 Graph Executor，最大化性能优化
- **功能扩展**：通过继承和组合支持多模态、MoE 等特殊场景

## 2. 各执行器功能详解

### 2.1 BaseExecutorImpl (基础执行器)

**文件位置**：`core/runtime/base_executor_impl.h/cpp`

**注册方式**：`REGISTER_EXECUTOR("llm", BaseExecutorImpl)` + `REGISTER_EXECUTOR("rec", RecExecutorImpl)`

**核心实现**：

```cpp
ModelOutput BaseExecutorImpl::run(const torch::Tensor& tokens,
                                  const torch::Tensor& positions,
                                  std::vector<KVCache>& kv_caches,
                                  const ModelInputParams& params) {
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}
```

**特点**：

- **Eager 模式执行**：直接调用模型的 `forward` 方法，无图优化
- **适用场景**：纯文本 LLM 模型、推荐模型 Rec、调试场景
- **性能特点**：简单直接，适合小 batch 或不需要 Graph Mode 的场景

**适用模型类型**：

- LLM（Large Language Model）：如 Qwen、DeepSeek、GLM 等纯文本模型
- Rec（Recommendation Model）：推荐系统模型

### 2.2 VlmExecutorImpl (多模态执行器)

**文件位置**：`core/runtime/vlm_executor_impl.h/cpp`

**注册方式**：`REGISTER_EXECUTOR("vlm", VlmExecutorImpl)`

**核心 Pipeline**：

```cpp
ModelOutput VlmExecutorImpl::run(const torch::Tensor& tokens,
                                  const torch::Tensor& positions,
                                  std::vector<KVCache>& kv_caches,
                                  const ModelInputParams& params) {
  // 1. 编码器缓存查询（避免重复编码）
  EncoderCacheLookupVisitor lookup(encoder_cache_.get());
  mm_data.foreach(lookup);

  // 2. 多模态数据编码
  MMDict embedding = encode(params);

  // 3. 嵌入向量注入到 token 序列
  params.embedding.input_embedding = 
      model_->get_input_embeddings(tokens, params);

  // 4. LLM 推理（可复用其他 Executor）
  if (llm_executor_) {
    return llm_executor_->run(tokens, positions, kv_caches, params);
  }
  return model_->forward(tokens, positions, kv_caches, params);
}
```

**关键特性**：

1. **编码器缓存（Encoder Cache）**
   - 缓存图像/视频特征，避免重复编码
   - 通过 `EncoderCacheLookupVisitor` 查询缓存
   - 通过 `EncoderCacheInsertVisitor` 插入新缓存

2. **多模态 Pipeline**
   ```
   图像输入 → EncoderCache → VisionEncoder → Embedding → LLMExecutor
                ↑                                              ↓
           Cache Lookup                                    LLM Forward
   ```

3. **Executor 复用设计**
   - 内部持有 `llm_executor_`，可使用 Graph Mode 优化
   - 根据 `enable_graph` 配置选择 BaseExecutor 或 GraphExecutor

**适用模型类型**：

- VLM（Vision-Language Model）：如 Qwen2-VL、Qwen3-VL、GLM4V、Oxygen-VLM 等

### 2.3 DiTExecutor (扩散模型执行器)

**文件位置**：`core/runtime/dit_executor.h/cpp`

**特点**：

- **独立接口设计**：不继承 `ExecutorImpl`，因为 DiT 模型无 KV Cache
- **专用 Forward 接口**：
  ```cpp
  DiTForwardOutput forward(const DiTForwardInput& input) {
    return model_->forward(input);
  }
  ```

**适用场景**：

- DiT（Diffusion Transformer）模型
- 图像生成任务（如 Stable Diffusion 3、Flux、Wan2.2）

**架构差异**：

- DiT 模型不需要 KV Cache，因此输入输出结构不同于 LLM
- 执行流程简化为：`prepare_inputs → forward → output`

### 2.4 CudaGraphExecutorImpl (CUDA Graph 优化)

**文件位置**：`core/runtime/cuda_graph_executor_impl.h/cpp`

**核心机制**：

```cpp
class CudaGraphExecutorImpl : public ExecutorImpl {
  // 持久化参数（避免重复分配）
  std::unique_ptr<CudaGraphPersistentParam> persistent_param_;

  // Graph 缓存（按 token 数分桶）
  absl::flat_hash_map<uint64_t, std::unique_ptr<CudaGraph>> graphs_;
};

class CudaGraph {
  at::cuda::CUDAGraph graph_;  // PyTorch CUDA Graph
  TorchMemPool* mem_pool_;     // 内存池管理临时张量

  bool capture(...);           // 捕获计算图
  ModelOutput replay(...);     // 重放捕获的图
};
```

**关键设计**：

1. **内存池（Memory Pool）**
   - 使用 CUDA Private MemPool 管理临时张量
   - 避免 Graph Capture 期间动态内存分配
   - 支持 Graph 间内存复用

2. **分桶策略（Bucket Strategy）**
   - 按 token 数预编译多个 Graph：`[1, 2, 4, 8, 16, 32, 64, 128, ...]`
   - 每个 bucket 对应一个独立 Graph 实例
   - 实际 token 数向上取整到最近的 bucket

3. **持久化缓冲（Persistent Buffer）**
   - 预分配输入/输出张量（tokens、positions、block_tables）
   - 避免 replay 前动态分配
   - Graph Capture 时使用固定地址

**Graph Capture 流程**：

```cpp
// 1. 准备持久化参数
auto graph_params = persistent_param_.update(tokens, k_cache, v_cache, positions, params, bucket_num_tokens);

// 2. 捕获计算图
graph_.capture_begin();
model_->forward(tokens, positions, kv_caches, graph_params);
graph_.capture_end();

// 3. 缓存 Graph 实例
graphs_[graph_key] = std::make_unique<CudaGraph>(...);
```

**Graph Replay 流程**：

```cpp
// 1. 更新持久化缓冲内容
persistent_param_.update(tokens, k_cache, v_cache, positions, params, actual_num_tokens);

// 2. 重放 Graph
auto output = graph_.replay();

// 3. 返回结果
return ModelOutput(output);
```

**性能提升**：

- 减少 kernel launch overhead：约 10-20%
- 避免重复内存分配：节省约 30% 内存碎片
- 提升吞吐与延迟稳定性

**适用硬件后端**：

- NVIDIA GPU（CUDA）

### 2.5 AclGraphExecutorImpl (NPU Graph 优化)

**文件位置**：`core/runtime/acl_graph_executor_impl.h/cpp`

**核心机制**：

```cpp
class AclGraphExecutorImpl : public ExecutorImpl {
  std::unique_ptr<GraphPersistentParam> persistent_param_;
  absl::flat_hash_map<uint64_t, std::unique_ptr<AclGraph>> graphs_;
};

class AclGraph {
  c10_npu::NPUGraph graph_;  // 华为 NPU Graph
  aclrtStream graph_stream_; // ACL 流
};
```

**ACL Graph Capture**：

```cpp
bool AclGraph::capture(...) {
  // 1. 同步设备
  torch::npu::synchronize();

  // 2. 获取 NPU Stream
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  // 3. 准备持久化参数
  auto graph_params = persistent_param_.update(...);

  // 4. 准备 Graph 元数据
  prepare_model_graph_metadata(model, positions, graph_params);

  // 5. 捕获计算图
  graph_.capture_begin(stream);
  model_->forward(...);
  graph_.capture_end();
}
```

**关键特性**：

- **华为 Ascend NPU 专用**：使用 `NPUGraph` API
- **NPUGraph MemPool**：类似 CUDA Graph，管理临时张量
- **分桶策略**：
  - 小 batch：`[1, 2, 4, 8]`
  - 大 batch：`[8, 16, 24, 32, 40, 48, ...]`（multiples of 8）

**适用硬件后端**：

- 华为 Ascend NPU（Atlas 系列）

### 2.6 DcuGraphExecutorImpl & MluGraphExecutorImpl

**DcuGraphExecutorImpl**：

- **海光 DCU**（AMD GPU 架构）专用
- 类似 CUDA Graph 实现，使用 DCU 特定 API

**MluGraphExecutorImpl**：

- **寒武纪 MLU**（Cambricon）专用
- 使用寒武纪提供的 Graph API

**设计差异**：

- 基本架构与 CUDA/NPU Graph Executor 相同
- 主要区别在于底层 Graph API 调用

### 2.7 EplbExecutor (专家负载均衡)

**文件位置**：`core/framework/eplb/eplb_executor.h/cpp`

**核心机制**：

```cpp
class EplbExecutor {
  void eplb_execute(const EplbInfo& eplb_info) {
    // 1. 根据 EplbInfo 准备专家权重
    prepare_expert_weight_async(layer_id, expert_ids, callback);

    // 2. 异步加载权重到 GPU/NPU
    callback(layer_id);
  }

 private:
  std::thread eplb_worker_;           // 异步任务线程
  std::queue<Task> tasks_;            // 任务队列
  std::mutex queue_mutex_;            // 队列锁
  std::condition_variable condition_; // 条件变量
};
```

**设计原理**：

1. **异步任务队列**
   - `std::thread eplb_worker_` 处理权重准备任务
   - 通过任务队列解耦权重加载与推理执行

2. **Layer-wise 准备**
   - 按 layer ID 分批加载 MoE 专家权重
   - 避免一次性加载所有权重导致内存爆炸

3. **专家选择策略**
   - 根据 `EplbInfo` 确定当前 step 需要的专家 ID
   - 动态加载激活专家的权重

**适用场景**：

- 大规模 MoE 模型（如 DeepSeek-V3、Qwen3-5-MoE）
- Expert Parallel Load Balancing

## 3. 执行器选择策略

### 3.1 工厂模式注册

**文件位置**：`core/runtime/executor_impl_factory.h/cpp`

**注册机制**：

```cpp
#define REGISTER_EXECUTOR(backend, class_type)                                 \
  namespace {                                                                  \
  bool class_type##_registered = []() -> bool {                                \
    return ExecutorImplFactory::get_instance().register_creator(               \
        backend,                                                               \
        [](CausalLM* model,                                                    \
           const ModelArgs& args,                                              \
           const torch::Device& device,                                        \
           const runtime::Options& options) -> std::unique_ptr<ExecutorImpl> { \
          return std::make_unique<class_type>(model, args, device, options);   \
        });                                                                    \
  }();                                                                         \
  }
```

**使用示例**：

```cpp
REGISTER_EXECUTOR("llm", BaseExecutorImpl);
REGISTER_EXECUTOR("vlm", VlmExecutorImpl);
REGISTER_EXECUTOR("rec", RecExecutorImpl);
```

### 3.2 选择逻辑

**工厂创建方法**：

```cpp
std::unique_ptr<ExecutorImpl> create_executor_impl(
    CausalLM* model,
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options,
    const std::string& backend) {

  auto& creators = creators_;
  auto it = creators.find(backend);
  if (it != creators.end()) {
    return it->second(model, args, device, options);
  }

  // 未找到则根据设备类型选择 Graph Executor
  return Device::create_graph_executor(model, args, device, options);
}
```

**选择策略**：

1. **根据 backend 类型**：
   - `"llm"` → BaseExecutorImpl 或 Graph Executor
   - `"vlm"` → VlmExecutorImpl（内部可复用 Graph Executor）
   - `"rec"` → RecExecutorImpl
   - `"dit"` → DiTExecutor（独立接口）

2. **根据设备类型**：
   - CUDA → CudaGraphExecutorImpl
   - NPU → AclGraphExecutorImpl
   - DCU → DcuGraphExecutorImpl
   - MLU → MluGraphExecutorImpl

3. **根据配置**：
   - `enable_graph=true` → Graph Executor
   - `enable_graph=false` → BaseExecutorImpl

## 4. 执行流程

### 4.1 LLM/VLM 执行流程

```
用户请求
   ↓
Worker (worker_impl.cpp)
   ↓
Executor.prepare_inputs(Batch)
   ↓
   ├─ VlmExecutor: EncoderCache → VisionEncoder
   └─ BaseExecutor/GraphExecutor: Prepare ForwardInput
   ↓
Executor.run(tokens, positions, kv_caches, params)
   ↓
   ├─ Eager Mode: model_->forward()
   ├─ CudaGraph: graph_.replay()
   └─ AclGraph: graph_.replay()
   ↓
ModelOutput (logits, hidden_states)
   ↓
Scheduler → Sampling → 返回结果
```

### 4.2 VLM 多模态执行细节

```cpp
// 1. 准备多模态输入
ForwardInput prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(num_decoding_tokens, 0, args, cp_size);
}

// 2. 多模态编码
ModelOutput run(...) {
  // 2.1 缓存查询
  EncoderCacheLookupVisitor lookup(encoder_cache_.get());
  mm_data.foreach(lookup);

  // 2.2 数据编码
  MMDict embedding = encode(params);

  // 2.3 嵌入注入
  params.embedding.input_embedding = model_->get_input_embeddings(tokens, params);

  // 2.4 LLM 推理
  return llm_executor_->run(tokens, positions, kv_caches, params);
}
```

### 4.3 Graph Mode 执行细节

```cpp
// 1. 请求分桶
uint32_t bucket_num_tokens = get_bucket_num_tokens(num_tokens);
uint64_t graph_key = get_graph_key(bucket_num_tokens, params);

// 2. Graph 缓存查询
auto& graph = graphs_[graph_key];
if (!graph) {
  // 2.1 未命中，执行 Capture
  graph = std::make_unique<AclGraph>(persistent_param_, device_index);
  graph->capture(model, options, tokens, positions, params, kv_caches, bucket_num_tokens);
}

// 3. 更新持久化缓冲
persistent_param_.update(tokens, k_cache, v_cache, positions, params, actual_num_tokens);

// 4. Graph Replay
return graph->replay(model, tokens, positions, kv_caches, params);
```

## 5. 执行器对比总结

| 执行器 | 适用模型 | 关键优化 | 设备支持 | 注册方式 |
|--------|----------|----------|----------|----------|
| BaseExecutorImpl | LLM/Rec | Eager 执行 | 全平台 | `REGISTER_EXECUTOR("llm")` |
| VlmExecutorImpl | VLM | 编码器缓存 + LLM 复用 | 全平台 | `REGISTER_EXECUTOR("vlm")` |
| DiTExecutor | DiT | 独立 Pipeline | 全平台 | 独立类 |
| CudaGraphExecutor | LLM | CUDA Graph + MemPool | NVIDIA GPU | 设备自动选择 |
| AclGraphExecutor | LLM | NPU Graph + MemPool | Ascend NPU | 设备自动选择 |
| DcuGraphExecutor | LLM | DCU Graph | 海光 DCU | 设备自动选择 |
| MluGraphExecutor | LLM | MLU Graph | 寒武纪 MLU | 设备自动选择 |
| EplbExecutor | MoE | 异步专家权重加载 | 全平台 | 独立模块 |

**核心设计思想**：

- **统一接口**：通过 `ExecutorImpl` 抽象基类定义标准接口
- **平台特化**：不同硬件后端使用专用 Graph Executor
- **内存优化**：Graph Capture + MemPool 减少动态分配
- **多模态支持**：VLM Pipeline + EncoderCache 避免重复编码
- **MoE 优化**：EPLB 异步加载大规模专家权重

## 6. 关键代码位置

| 模块 | 文件路径 |
|------|----------|
| Executor 抽象基类 | `xllm/core/runtime/executor_impl.h` |
| Executor 封装类 | `xllm/core/runtime/executor.h` |
| BaseExecutor | `xllm/core/runtime/base_executor_impl.h/cpp` |
| VlmExecutor | `xllm/core/runtime/vlm_executor_impl.h/cpp` |
| DiTExecutor | `xllm/core/runtime/dit_executor.h/cpp` |
| CudaGraphExecutor | `xllm/core/runtime/cuda_graph_executor_impl.h/cpp` |
| AclGraphExecutor | `xllm/core/runtime/acl_graph_executor_impl.h/cpp` |
| ExecutorFactory | `xllm/core/runtime/executor_impl_factory.h/cpp` |
| EplbExecutor | `xllm/core/framework/eplb/eplb_executor.h/cpp` |

## 7. 扩展阅读

- [Graph Mode 设计文档](graph_mode_design.md)：深入了解 Graph Capture/Replay 原理
- [代码结构文档](../dev_guide/code_arch.md)：了解整体代码组织
- [多模态功能文档](../features/multimodal.md)：了解 VLM 模型使用方式
- [EPLB 功能文档](../features/eplb.md)：了解 MoE 专家负载均衡机制