# xLLM Model 软件架构详解

## 1. Model 类型体系概览

xLLM 的 Model 层采用多层继承 + CRTP 模板设计，实现了高度的可扩展性和代码复用。

### 1.1 类型层次图

```
                    ┌─────────────────────────────────────┐
                    │     torch::nn::Module (LibTorch)     │
                    └──────────────────┬──────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
    ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
    │  CausalLM   │            │  DiTModel   │            │ RecCausalLM │
    │ (LLM基类)   │            │ (DiT基类)   │            │ (推荐基类)  │
    └──────┬──────┘            └──────┬──────┘            └──────┬──────┘
           │                           │                           │
           │                           │                           │
           ▼                           ▼                           ▼
    ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
    │CausalLMImpl │            │DiTModelImpl │            │RecCausalLM  │
    │ <Model>     │            │ <Model>     │            │Impl<Model>  │
    │(模板包装器) │            │(模板包装器) │            │(模板包装器) │
    └──────┬──────┘            └──────┬──────┘            └──────┬──────┘
           │                           │                           │
           ├─ CausalVLM ───────────────┤                           │
           │  (VLM基类)                │                           │
           ▼                           │                           │
    ┌─────────────┐                    │                           │
    │CausalVLMImpl│                    │                           │
    │ <Model>     │                    │                           │
    │(VLM包装器)  │                    │                           │
    └──────┬──────┘                    │                           │
           │                           │                           │
           │                           │                           │
           ▼                           ▼                           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   具体模型实现层 (CRTP)                          │
    ├──────────────┬──────────────┬──────────────┬──────────────────┤
    │LlmForCausalLM│VlmForCondGen │DiTImplBase   │RecForCausalLM    │
    │ImplBase      │ImplBase      │              │ImplBase          │
    │<LlmModelType>│<VlmModelType>│              │<RecModelType>    │
    └──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────────┘
           │              │              │              │
           ▼              ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │LlamaForCausal│ │Qwen2_VLFor   │ │FluxPipeline  │ │OneRecFor     │
    │LM            │ │CondGen       │ │              │ │CondGen       │
    │              │ │              │ │              │ │              │
    └─ model_:     │ └─ model_:     │ └─ transformer│ └─ model_:     │
    │  LlamaModel  │ │  Qwen2VLModel│ │  layers_[]   │ │  OneRecModel │
    └─ lm_head_    │ └─ lm_head_    │ └              │ └─ lm_head_   │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │              │              │              │
           ▼              ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │LlmModelImpl  │ │VlmModelImpl  │ │              │ │RecModelImpl  │
    │Base          │ │Base          │ │              │ │Base          │
    │<DecoderLayer>│ │<DecoderLayer>│ │              │ │              │
    └              │ └              │ │              │ └              │
    ├ embed_tokens_│ ├ embed_tokens_│ │              │ ├ embed_tokens_│
    ├ layers_[]    │ ├ layers_[]    │ │              │ ├ layers_[]   │
    ├ norm_        │ ├ norm_        │ │              │ ├ norm_       │
    └──────┬───────┘ └──────┬───────┘ │              │ └──────┬───────┘
           │              │          │              │          │
           ▼              ▼          │              │          ▼
    ┌──────────────┐ ┌──────────────┐│      ┌──────────────┐
    │DecoderLayer  │ │DecoderLayer  ││      │DecoderLayer  │
    │ImplBase      │ │ImplBase      ││      │ImplBase      │
    │<AttnImpl>    │ │<AttnImpl>    ││      │              │
    ├ decoder_layer│ ├ decoder_layer││      ├ decoder_layer│
    ├ block_copy_  │ ├ block_copy_  ││      └              │
    └──────┬───────┘ └──────┬───────┘│      └──────┬───────┘
           │              │         │              │
           ▼              ▼         │              ▼
    ┌──────────────┐ ┌──────────────┐      ┌──────────────┐
    │NpuQwen2      │ │NpuQwen2VL    │      │具体Decoder   │
    │DecoderLayer  │ │DecoderLayer  │      │Layer Impl    │
    │              │ │              │      │              │
    ├ self_attn_   │ ├ self_attn_   │      ├ self_attn_   │
    ├ mlp_         │ ├ mlp_         │      ├ mlp_         │
    ├ input_layern │ ├ input_layern │      ├ input_layern │
    ├ post_attn_ln │ ├ post_attn_ln │      ├ post_attn_ln │
    └──────────────┘ └──────────────┘      └──────────────┘
```

---

## 2. 核心类详解

### 2.1 CausalLM (LLM 基类)

**职责**：定义 LLM 模型的核心接口

```cpp
// core/framework/model/causal_lm.h
class CausalLM : public torch::nn::Module {
 public:
  // ★ 核心 forward 方法
  virtual ModelOutput forward(
      const torch::Tensor& tokens,        // [num_tokens]
      const torch::Tensor& positions,     // [num_tokens]
      std::vector<KVCache>& kv_caches,    // 每个 layer 的 KV cache
      const ModelInputParams& parameters) = 0;
  
  // 计算 logits
  virtual torch::Tensor logits(
      const torch::Tensor& hidden_states,   // [num_tokens, hidden_size]
      const torch::Tensor& seleted_idxes) = 0;  // [num_tokens]
  
  // Pooler（用于推荐模型等）
  virtual torch::Tensor pooler(
      const torch::Tensor& hidden_states,
      const torch::Tensor& seleted_idxes);
  
  // 加载模型权重
  virtual void load_model(std::unique_ptr<ModelLoader> loader) = 0;
  
  // 设备管理
  virtual torch::Device device() const = 0;
  virtual const torch::TensorOptions& options() const = 0;
  
  // Expert parallel load balancing (MoE)
  virtual void prepare_expert_weight(int32_t layer_id,
                                      const std::vector<int32_t>& expert_ids) = 0;
  virtual void update_expert_weight(int32_t layer_id) = 0;
  
  // 权重管理（用于 rolling load 等）
  virtual void lazy_load_model(std::unique_ptr<ModelLoader> loader);
  virtual void free_model_weights();
  virtual void reload_model_weights();
  virtual void reload_model_weights_from_device();
  
  // NPU 特化接口
#if defined(USE_NPU)
  virtual layer::NpuLmHead get_npu_lm_head();
  virtual void set_npu_lm_head(layer::NpuLmHead& head);
  virtual layer::NpuWordEmbedding get_npu_word_embedding();
  virtual void set_npu_word_embedding(layer::NpuWordEmbedding& embedding);
  virtual bool init_or_refresh_rolling_runtime(...);
#endif
  
  // CUDA/MLU 特化接口
  virtual layer::LmHead get_lm_head();
  virtual void set_lm_head(layer::LmHead& head);
  virtual layer::WordEmbedding get_word_embedding();
  virtual void set_word_embedding(layer::WordEmbedding& embedding);
};
```

### 2.2 CausalLMImpl<Model> (模板包装器)

**职责**：通过 CRTP 模板包装具体模型实现，使用 traits 检测机制选择性地调用方法

```cpp
// core/framework/model/causal_lm.h:168-366
template <typename Model>
class CausalLMImpl : public CausalLM {
 public:
  CausalLMImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}
  
  // ★ 核心 forward
  ModelOutput forward(...) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }
  
  // logits 计算
  torch::Tensor logits(...) override {
    return model_->logits(hidden_states, seleted_idxes);
  }
  
  // Traits 检测机制：选择性调用
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden) override {
    // 检测 Model 是否有 logits(hidden, idx, out_hidden) 方法
    if constexpr (detail::has_logits_with_hidden<Model>::value) {
      return model_->logits(hidden_states, seleted_idxes, out_hidden);
    } else {
      return CausalLM::logits(hidden_states, seleted_idxes, out_hidden);
    }
  }
  
  void lazy_load_model(std::unique_ptr<ModelLoader> loader) override {
    if constexpr (detail::has_lazy_load_model<Model>::value) {
      model_->lazy_load_model(std::move(loader));
    } else {
      CausalLM::lazy_load_model(std::move(loader));
    }
  }
  
  // ... 其他 traits 检测
  
 private:
  Model model_;                        // 具体模型实例（如 LlamaForCausalLM）
  torch::TensorOptions options_;       // Tensor 配置
};
```

**Traits 检测机制详解**：

```cpp
// core/framework/model/model_traits.h
namespace detail {

// 检测 Model 是否有 logits(hidden, idx, out_hidden) 方法
template <typename T, typename = void>
struct has_logits_with_hidden : std::false_type {};

template <typename T>
struct has_logits_with_hidden<T,
    std::void_t<decltype(std::declval<T>()->logits(
        std::declval<const torch::Tensor&>(),
        std::declval<const torch::Tensor&>(),
        std::declval<torch::Tensor&>()))>>
    : std::true_type {};

// 检测其他方法
template <typename T, typename = void>
struct has_get_lm_head : std::false_type {};
template <typename T>
struct has_get_lm_head<T,
    std::void_t<decltype(std::declval<T>()->get_lm_head())>>
    : std::true_type {};

// ... 更多 traits
}
```

### 2.3 CausalVLM (VLM 基类)

**职责**：扩展 CausalLM，增加多模态处理接口

```cpp
// core/framework/model/causal_vlm.h
class CausalVLM : public CausalLM {
 public:
  // ★ 多模态编码
  virtual MMDict encode(const ModelInputParams& parameters) = 0;
  
  // 获取输入 embeddings（包含 image embeddings）
  virtual torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelInputParams& input_params) = 0;
};

template <typename Model>
class CausalVLMImpl : public CausalVLM {
 public:
  MMDict encode(const ModelInputParams& parameters) override {
    return model_->get_multimodal_embeddings(parameters);
  }
  
  torch::Tensor get_input_embeddings(...) override {
    return model_->get_input_embeddings(input_ids, input_params);
  }
  
  // 继承 CausalLM 的 forward、logits 等方法
  ModelOutput forward(...) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }
};
```

### 2.4 RecCausalLM (推荐模型基类)

**职责**：推荐场景特化的 CausalLM

```cpp
// core/framework/model/rec_causal_lm.h
class RecCausalLM : public CausalLM {
 public:
  ~RecCausalLM() override = default;
  // 继承 CausalLM 所有接口
  // 推荐模型通常不需要 sampling，只计算 logits/pooler
};

template <typename Model>
class RecCausalLMImpl : public RecCausalLM {
  // 与 CausalLMImpl 类似的模板包装器
};
```

---

## 3. 具体模型实现层 (CRTP)

### 3.1 LlmForCausalLMImplBase<LlmModelType>

**职责**：LLM "ForCausalLM" 层，包含 Transformer + LM Head

```cpp
// models/llm/llm_model_base.h (CUDA/MLU)
// models/llm/npu/llm_model_base.h (NPU)
template <typename LlmModelType>
class LlmForCausalLMImplBase : public torch::nn::Module {
 public:
  LlmForCausalLMImplBase(const ModelContext& context) {
    tie_word_embeddings = context.get_model_args().tie_word_embeddings();
    
    // 注册子模块
    model_ = register_module("model", LlmModelType(context));
    lm_head_ = register_module("lm_head", layer::LmHead(context));
    // NPU: npu_lm_head_ = register_module("npu_lm_head", layer::NpuLmHead(context));
  }
  
  // ★ Forward: Transformer → hidden states
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    return model_(tokens, positions, kv_caches, input_params);
  }
  
  // ★ Logits: hidden states → vocab logits
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);  // 选择特定位置的 hidden
    }
    return lm_head_(h);  // LM Head: [num_seqs, vocab_size]
  }
  
  // Pooler: 用于推荐等场景
  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return h;
  }
  
  // 加载权重
  void load_model(std::unique_ptr<ModelLoader> loader,
                   std::string prefix = "model.") override {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(
          state_dict->get_dict_with_prefix({"model.language_model.",
                                             "language_model.model.",
                                             prefix, "model.", ""}));
      
      if (tie_word_embeddings) {
        // tie_word_embeddings: lm_head 权重与 embed_tokens 共享
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix({prefix + "embed_tokens.",
                                             "embed_tokens.", "embed."}));
      } else {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix({"lm_head.", "model.lm_head.",
                                             "model.head.", "head.", ...}));
      }
    }
  }
  
  // LM Head / Word Embedding 管理
  layer::LmHead get_lm_head() override { return lm_head_; }
  void set_lm_head(layer::LmHead& head) override { lm_head_ = head; }
  layer::WordEmbedding get_word_embedding() override {
    return model_->get_word_embedding();
  }
  void set_word_embedding(layer::WordEmbedding& embedding) override {
    model_->set_word_embedding(embedding);
  }
  
 protected:
  LlmModelType model_;              // Transformer (如 LlamaModel, Qwen2Model)
  layer::LmHead lm_head_;           // LM Head (CUDA/MLU)
  // layer::NpuLmHead npu_lm_head_; // LM Head (NPU)
  
  bool tie_word_embeddings{false};  // 是否共享 embedding 权重
  bool keep_host_weights{false};    // 是否保留 host 权重（rolling load）
};
```

### 3.2 LlmModelImplBase<DecoderLayerType>

**职责**：Transformer 主体（Embedding → DecoderLayers → Norm）

```cpp
// models/llm/llm_model_base.h
template <typename DecoderLayerType>
class LlmModelImplBase : public torch::nn::Module {
 public:
  LlmModelImplBase(const std::string& model_type, const ModelArgs& args)
      : model_args_(args), model_type_(model_type) {
    // 初始化
    InterruptionBus::get_instance().subscribe([this](bool interrupted) {
      this->layer_forward_interrupted_ = interrupted;
    });
  }
  
  // ★ 核心 forward: Embedding → Layers → Norm
  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }
    
    // 1. Embedding
    torch::Tensor h;
    if (input_params.embedding.input_embedding.defined()) {
      h = input_params.embedding.input_embedding;  // 使用外部 embedding
    } else {
      h = embed_tokens_(tokens);  // Word embedding
    }
    
    // 2. Attention metadata 构建
    if (!input_params.attn_metadata) {
      input_params.attn_metadata = 
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(input_params, ...));
    }
    
    // 3. Decoder layers forward
    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      h = layers_[i](h, residual, positions, 
                     *input_params.attn_metadata, 
                     kv_caches[i], input_params);
      
      // 中断检测（用于 request cancellation）
      if (layer_forward_interrupted_) {
        return ModelOutput();
      }
    }
    
    // 4. Final norm
    auto [hidden_states, residual_out] = norm_(h, residual);
    return ModelOutput(hidden_states, residual_out);
  }
  
  // 加载权重
  void load_state_dict(const StateDict& state_dict) override {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }
  
  // Embedding 管理
  layer::WordEmbedding get_word_embedding() override { return embed_tokens_; }
  void set_word_embedding(layer::WordEmbedding& embedding) override {
    embed_tokens_ = embedding;
  }
  
 protected:
  ModelArgs model_args_;
  layer::WordEmbedding embed_tokens_;     // Word embedding
  layer::RMSNorm norm_;                   // Final normalization
  std::vector<DecoderLayerType> layers_;  // Decoder layers
  
  bool layer_forward_interrupted_ = false;
};

// NPU 版本（有额外组件）
template <typename DecoderLayerType>
class LlmModelImplBase : public torch::nn::Module {
 protected:
  layer::NpuWordEmbedding npu_embed_tokens_;
  layer::NpuRMSNorm norm_;
  layer::NpuPosEmbedding atb_pos_emb_;
  torch::Tensor cos_sin_;
  layer::AttentionMask attn_mask_;
  std::vector<DecoderLayerType> layers_;
  RollingLoadManager* rolling_mgr_;
};
```

### 3.3 LlmDecoderLayerImplBase<AttnImpl>

**职责**：Decoder Layer（BlockCopy → Attention → MLP）

```cpp
// models/llm/npu/llm_model_base.h
template <typename DecoderType>
class LlmDecoderLayerImplBase : public torch::nn::Module {
 public:
  LlmDecoderLayerImplBase(const ModelContext& context, const int32_t layer_id)
      : layer_id_(layer_id) {
    decoder_layer_ = register_module("decoder_layer", DecoderType(context));
    block_copy_ = register_module("block_copy", layer::NpuBlockCopy(context));
    decoder_layer_->set_layer_id(layer_id_);
  }
  
  // ★ Forward: BlockCopy → DecoderLayer
  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        aclrtEvent* event,
                        std::atomic<bool>* event_flag) {
    // 1. KV Cache block copy（用于 prefix cache）
    if (input_params.block_copy.src_block_indices.numel() > 0) {
      block_copy_(kv_cache.get_k_cache(),
                  kv_cache.get_v_cache(),
                  input_params.block_copy.src_block_indices,
                  input_params.block_copy.dst_block_indices, ...);
    }
    
    // 2. Decoder layer forward (attention + mlp)
    return decoder_layer_(x, cos_pos, sin_pos, attn_mask, 
                          kv_cache, input_params, event, event_flag, layer_id_);
  }
  
  void load_state_dict(const StateDict& state_dict) override {
    decoder_layer_->load_state_dict(state_dict);
  }
  
 private:
  DecoderType decoder_layer_;    // 具体的 Decoder Layer（如 NpuQwen2DecoderLayer）
  layer::NpuBlockCopy block_copy_; // KV cache block copy
  int32_t layer_id_;
};
```

---

## 4. 完整模型实例分析（Llama/Qwen2）

### 4.1 LlamaForCausalLM 完整结构

```cpp
// models/llm/npu/llama.h

// ==================== 层级 4: Decoder Layer ====================
class NpuLlamaDecoderLayer : public torch::nn::Module {
  // Self-attention + MLP + Norms
 private:
  NpuSelfAttention self_attn_;
  NpuMLP mlp_;
  NpuRMSNorm input_layernorm_;
  NpuRMSNorm post_attention_layernorm_;
};
TORCH_MODULE(NpuLlamaDecoderLayer);

// ==================== 层级 3: Decoder Layer Wrapper ====================
class LlamaDecoderLayerImpl 
    : public LlmDecoderLayerImplBase<NpuLlamaDecoderLayer> {
  // 继承 LlmDecoderLayerImplBase，添加 block_copy
};
TORCH_MODULE(LlamaDecoderLayer);

// ==================== 层级 2: Transformer ====================
class LlamaModelImpl : public torch::nn::Module {
 public:
  LlamaModelImpl(const ModelContext& context) {
    // 注册子模块
    npu_embed_tokens_ = register_module("npu_embed_tokens", 
                                        layer::NpuWordEmbedding(context));
    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    blocks_ = register_module("layers", torch::nn::ModuleList());
    
    // 创建 decoder layers
    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = LlamaDecoderLayer(context);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    
    // Rotary embedding
    cos_pos_, sin_pos_ = get_llama_rotary_embedding(...);
    attn_mask_ = layer::AttentionMask(...);
  }
  
  ModelOutput forward(...) {
    // Embedding
    torch::Tensor h = npu_embed_tokens_(tokens, 0);
    
    // RoPE
    auto cos_pos = cos_pos_.index_select(0, positions);
    auto sin_pos = sin_pos_.index_select(0, positions);
    
    // Attention mask
    auto attn_mask = attn_mask_.get_attn_mask(max_seq_len_, ...);
    
    // Decoder layers
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i](h, cos_pos, sin_pos, attn_mask, kv_caches[i], ...);
    }
    
    // Final norm
    auto hidden_states = norm_(h, 0);
    return ModelOutput(hidden_states);
  }
  
 private:
  layer::NpuWordEmbedding npu_embed_tokens_;
  layer::NpuRMSNorm norm_;
  std::vector<LlamaDecoderLayer> layers_;
  torch::nn::ModuleList blocks_;
  
  torch::Tensor cos_pos_, sin_pos_;
  layer::AttentionMask attn_mask_;
};
TORCH_MODULE(LlamaModel);

// ==================== 层级 1: ForCausalLM ====================
class LlamaForCausalLMImpl 
    : public LlmForCausalLMImplBase<LlamaModel> {
 public:
  LlamaForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<LlamaModel>(context) {}
  
  // 继承 LlmForCausalLMImplBase:
  //   forward() → model_->forward()
  //   logits() → npu_lm_head_->forward()
};
TORCH_MODULE(LlamaForCausalLM);

// ==================== 注册 ====================
REGISTER_CAUSAL_MODEL(llama, LlamaForCausalLM);
REGISTER_MODEL_ARGS(llama, [&] {
  LOAD_ARG_OR(model_type, "model_type", "llama");
  LOAD_ARG_OR(hidden_size, "hidden_size", 8192);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 80);
  // ... 更多参数
});
```

**完整类关系图（Llama）**：

```
CausalLMImpl<LlamaForCausalLM>
  └─ model_: LlamaForCausalLM
       ├─ model_: LlamaModel (Transformer)
       │    ├─ npu_embed_tokens_: NpuWordEmbedding
       │    ├─ layers_[]: LlamaDecoderLayer
       │    │    └─ decoder_layer_: NpuLlamaDecoderLayer
       │    │         ├─ self_attn_: NpuSelfAttention
       │    │         ├─ mlp_: NpuMLP
       │    │         ├─ input_layernorm_: NpuRMSNorm
       │    │         └─ post_attention_layernorm_: NpuRMSNorm
       │    ├─ norm_: NpuRMSNorm
       │    ├─ cos_pos_, sin_pos_: Rotary embedding
       │    └─ attn_mask_: AttentionMask
       └─ npu_lm_head_: NpuLmHead
```

### 4.2 Qwen2ForCausalLM 完整结构

```cpp
// models/llm/npu/qwen2.h

// ==================== 层级 4: Decoder Layer ====================
class NpuQwen2DecoderLayer : public torch::nn::Module {
 private:
  NpuQwen2SelfAttention self_attn_;
  NpuQwen2MLP mlp_;
  NpuRMSNorm input_layernorm_;
  NpuRMSNorm post_attention_layernorm_;
};
TORCH_MODULE(NpuQwen2DecoderLayer);

// ==================== 层级 3: Decoder Layer Wrapper ====================
class QWen2DecoderLayerImpl 
    : public LlmDecoderLayerImplBase<NpuQwen2DecoderLayer> {
  // 继承 LlmDecoderLayerImplBase
};
TORCH_MODULE(QWen2DecoderLayer);

// ==================== 层级 2: Transformer ====================
class QWen2ModelImpl : public LlmModelImplBase<QWen2DecoderLayer> {
 public:
  QWen2ModelImpl(const ModelContext& context)
      : LlmModelImplBase<QWen2DecoderLayer>("qwen2", context.get_model_args()) {
    blocks_ = register_module("layers", torch::nn::ModuleList());
    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    npu_embed_tokens_ = register_module("npu_embed_tokens", 
                                        layer::NpuWordEmbedding(context));
    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = rotary::get_concat_rotary_embedding(...);
    attn_mask_ = layer::AttentionMask(...);
    
    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = QWen2DecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }
};
TORCH_MODULE(QWen2Model);

// ==================== 层级 1: ForCausalLM ====================
class QWen2ForCausalLMImpl : public LlmForCausalLMImplBase<QWen2Model> {
 public:
  QWen2ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<QWen2Model>(context) {}
};
TORCH_MODULE(QWen2ForCausalLM);

// ==================== 注册 ====================
REGISTER_CAUSAL_MODEL(qwen2, QWen2ForCausalLM);
REGISTER_MODEL_ARGS(qwen2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen2");
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  // ... 更多参数
});
```

---

## 5. Model 注册机制详解

### 5.1 注册宏定义

```cpp
// models/model_registry.h:150-311

// ==================== CausalLM 注册 ====================
#define REGISTER_CAUSAL_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass) \
  const bool VarName##_registered = []() {                                 \
    ModelRegistry::register_causallm_factory(                              \
        #ModelType, [](const ModelContext& context) {                      \
          ModelClass model(context);  /* 创建具体模型实例 */                \
          model->eval();              /* 设置为 eval mode */                \
          /* 包装为 CausalLMImpl<ModelClass> */                            \
          return std::make_unique<xllm::CausalLMImpl<ModelClass>>(         \
              std::move(model), context.get_tensor_options());             \
        });                                                                \
    return true;                                                           \
  }()

#define REGISTER_CAUSAL_MODEL(ModelType, ModelClass) \
  REGISTER_CAUSAL_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

// ==================== VLM 注册 ====================
#define REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass) \
  const bool VarName##_registered = []() {                                     \
    ModelRegistry::register_causalvlm_factory(                                 \
        #ModelType, [](const ModelContext& context) {                          \
          ModelClass model(context);                                           \
          model->eval();                                                       \
          return std::make_unique<xllm::CausalVLMImpl<ModelClass>>(            \
              std::move(model), context.get_tensor_options());                 \
        });                                                                    \
    return true;                                                               \
  }()

#define REGISTER_CAUSAL_VLM_MODEL(ModelType, ModelClass) \
  REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

// ==================== RecModel 注册 ====================
#define REGISTER_REC_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass) \
  const bool VarName##_rec_registered = []() {                          \
    ModelRegistry::register_rec_model_factory(                          \
        #ModelType, [](const ModelContext& context) {                   \
          ModelClass model(context);                                    \
          model->eval();                                                \
          return std::make_unique<xllm::RecCausalLMImpl<ModelClass>>(   \
              std::move(model), context.get_tensor_options());          \
        });                                                             \
    return true;                                                        \
  }()

#define REGISTER_REC_MODEL(ModelType, ModelClass) \
  REGISTER_REC_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

// ==================== DiT Model 注册 ====================
#define REGISTER_DIT_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass) \
  const bool VarName##_registered = []() {                              \
    ModelRegistry::register_dit_model_factory(                          \
        #ModelType, [](const DiTModelContext& context) {                \
          ModelClass model(context);                                    \
          model->eval();                                                \
          return std::make_unique<xllm::DiTModelImpl<ModelClass>>(      \
              std::move(model), context.get_tensor_options());          \
        });                                                             \
    return true;                                                        \
  }()

#define REGISTER_DIT_MODEL(ModelType, ModelClass) \
  REGISTER_DIT_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

// ==================== Model Args 注册 ====================
#define REGISTER_MODEL_ARGS_WITH_VARNAME(VarName, ModelType, ...)       \
  REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(                              \
      VarName, ModelType, [](const JsonReader& json, ModelArgs* args) { \
        UNUSED_PARAMETER(json);                                         \
        UNUSED_PARAMETER(args);                                         \
        __VA_ARGS__();  /* 执行 lambda body */                          \
        return true;                                                    \
      })

#define REGISTER_MODEL_ARGS(ModelType, ...) \
  REGISTER_MODEL_ARGS_WITH_VARNAME(ModelType, ModelType, __VA_ARGS__)
```

### 5.2 ModelRegistry 单例

```cpp
// models/model_registry.h:76-129
class ModelRegistry {
 public:
  static ModelRegistry* get_instance();
  
  // 注册工厂函数
  static void register_causallm_factory(const std::string& name,
                                        CausalLMFactory factory);
  static void register_causalvlm_factory(const std::string& name,
                                         CausalVLMFactory factory);
  static void register_rec_model_factory(const std::string& name,
                                         RecModelFactory factory);
  static void register_dit_model_factory(const std::string& name,
                                         DiTModelFactory factory);
  
  // 注册参数加载器
  static void register_model_args_loader(const std::string& name,
                                         ModelArgsLoader loader);
  static void register_quant_args_loader(const std::string& name,
                                         QuantArgsLoader loader);
  static void register_tokenizer_args_loader(const std::string& name,
                                              TokenizerArgsLoader loader);
  
  // 获取工厂函数
  static CausalLMFactory get_causallm_factory(const std::string& name);
  static CausalVLMFactory get_causalvlm_factory(const std::string& name);
  static RecModelFactory get_rec_model_factory(const std::string& name);
  static DiTModelFactory get_dit_model_factory(const std::string& name);
  
  // 获取参数加载器
  static ModelArgsLoader get_model_args_loader(const std::string& name);
  
 private:
  std::unordered_map<std::string, ModelMeta> model_registry_;
  std::unordered_map<std::string, std::string> model_backend_;
};

// ModelMeta 结构（每个模型注册的所有信息）
struct ModelMeta {
  CausalLMFactory causal_lm_factory;
  RecModelFactory rec_model_factory;
  CausalVLMFactory causal_vlm_factory;
  DiTModelFactory dit_model_factory;
  MultimodalProcessorFactory multimodal_processor_factory;
  ModelArgsLoader model_args_loader;
  QuantArgsLoader quant_args_loader;
  TokenizerArgsLoader tokenizer_args_loader;
};
```

### 5.3 模型创建流程

```cpp
// models/model_registry.cpp:334-353
std::unique_ptr<CausalLM> create_llm_model(const ModelContext& context) {
  // 1. 解析 model_type
  std::string resolved_name;
  if (!resolve_model_registration_name(context.get_model_args().model_type(),
                                       &resolved_name)) {
    return nullptr;
  }
  
  // 2. 从 registry 获取工厂函数
  auto factory = ModelRegistry::get_causallm_factory(resolved_name);
  if (!factory) {
    LOG(ERROR) << "Unsupported model type: " << context.get_model_args().model_type();
    return nullptr;
  }
  
  // 3. 调用工厂函数创建模型
  return factory(context);  // 返回 CausalLMImpl<ModelClass>
}

// Worker 创建模型
// llm_worker_impl.cpp:84
bool LLMWorkerImpl::init_model(ModelContext& context) {
  model_ = create_llm_model(context);  // 通过工厂创建
  CHECK(model_ != nullptr) << "Failed to create model.";
  
  model_executor_ = std::make_unique<Executor>(model_.get(), ...);
  return true;
}
```

**完整注册流程图**：

```
┌──────────────────────────────────────────────────────────────┐
│         静态注册（编译时自动执行）                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  文件: models/llm/npu/llama.h                                │
│                                                              │
│  REGISTER_CAUSAL_MODEL(llama, LlamaForCausalLM);             │
│  ↓                                                           │
│  const bool llama_registered = []() {                        │
│    ModelRegistry::register_causallm_factory(                 │
│      "llama",                                                │
│      [](const ModelContext& context) {                       │
│        // 1. 创建 LlamaForCausalLM                           │
│        LlamaForCausalLM model(context);                      │
│        model->eval();                                        │
│                                                              │
│        // 2. 包装为 CausalLMImpl<LlamaForCausalLM>           │
│        return std::make_unique<CausalLMImpl<...>>(           │
│            std::move(model), options);                       │
│      });                                                     │
│    return true;                                              │
│  }()                                                         │
│                                                              │
│  REGISTER_MODEL_ARGS(llama, [&] {                            │
│    LOAD_ARG_OR(hidden_size, "hidden_size", 8192);            │
│    LOAD_ARG_OR(n_layers, "num_hidden_layers", 80);           │
│    ...                                                       │
│  });                                                         │
│                                                              │
│  ↓                                                           │
│  ModelRegistry::model_registry_["llama"] = {                 │
│    causal_lm_factory = factory_lambda,                       │
│    model_args_loader = args_loader_lambda,                   │
│  };                                                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│         模型创建（运行时）                                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Worker::init_model(ModelContext)                         │
│     ↓                                                        │
│  2. create_llm_model(context)                                │
│     ↓                                                        │
│  3. resolve_model_registration_name("llama") → "llama"       │
│     ↓                                                        │
│  4. factory = ModelRegistry::get_causallm_factory("llama")   │
│     ↓                                                        │
│  5. factory(context)                                         │
│     ↓                                                        │
│  6. LlamaForCausalLM model(context)                          │
│     ├─ LlamaModel(context)                                   │
│     │   ├─ NpuWordEmbedding                                  │
│     │   ├─ LlamaDecoderLayer[] (N layers)                    │
│     │   │   └─ NpuLlamaDecoderLayer                          │
│     │   │       ├─ NpuSelfAttention                          │
│     │   │       ├─ NpuMLP                                    │
│     │   │       └─ NpuRMSNorm[]                              │
│     │   └─ NpuRMSNorm                                        │
│     └─ NpuLmHead                                             │
│     ↓                                                        │
│  7. CausalLMImpl<LlamaForCausalLM>(model, options)           │
│     ↓                                                        │
│  8. return std::unique_ptr<CausalLM>                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. 参数加载机制

### 6.1 LOAD_ARG 宏详解

```cpp
// models/model_registry.h:282-309

// 从 JSON 加载可选参数（不存在则保持默认）
#define LOAD_ARG(arg_name, json_name)                          \
  [&] {                                                        \
    auto value = args->arg_name();                             \
    using value_type = remove_optional_t<decltype(value)>;     \
    if (auto data_value = json.value<value_type>(json_name)) { \
      args->arg_name() = data_value.value();                   \
    }                                                          \
  }()

// 从 JSON 加载参数（不存在则使用默认值）
#define LOAD_ARG_OR(arg_name, json_name, default_value)                     \
  [&] {                                                                     \
    auto value = args->arg_name();                                          \
    using value_type = remove_optional_t<decltype(value)>;                  \
    args->arg_name() = json.value_or<value_type>(json_name, default_value); \
  }()

// 从 JSON 加载参数（不存在则执行 lambda 计算默认值）
#define LOAD_ARG_OR_FUNC(arg_name, json_name, ...)             \
  [&] {                                                        \
    auto value = args->arg_name();                             \
    using value_type = remove_optional_t<decltype(value)>;     \
    if (auto data_value = json.value<value_type>(json_name)) { \
      args->arg_name() = data_value.value();                   \
    } else {                                                   \
      args->arg_name() = __VA_ARGS__();                        \
    }                                                          \
  }()

// 直接设置参数值
#define SET_ARG(arg_name, value) [&] { args->arg_name() = value; }()
```

### 6.2 参数加载示例

```cpp
// models/llm/npu/qwen2.h:90-119
REGISTER_MODEL_ARGS(qwen2, [&] {
  // 加载必需参数
  LOAD_ARG_OR(model_type, "model_type", "qwen2");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  
  // 加载可选参数（不存在则使用默认值）
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  
  // 加载参数并计算默认值
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  
  // 直接设置参数
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});
```

---

## 7. Model 类型对比表

| Model 类型 | 基类 | 包装器模板 | 具体实现基类 | 用途 |
|----------|------|----------|------------|------|
| **CausalLM** | CausalLM | CausalLMImpl<Model> | LlmForCausalLMImplBase | 标准 LLM（Llama, Qwen2, DeepSeek） |
| **CausalVLM** | CausalVLM (继承 CausalLM) | CausalVLMImpl<Model> | VlmForCondGenImplBase | 视觉语言模型（Qwen2-VL, Qwen3-VL） |
| **RecCausalLM** | RecCausalLM (继承 CausalLM) | RecCausalLMImpl<Model> | RecForCausalLMImplBase | 推荐模型（OneRec） |
| **DiTModel** | DiTModel | DiTModelImpl<Model> | DiTImplBase | Diffusion Transformer（Flux） |

---

## 8. 设计模式总结

### 8.1 CRTP（Curiously Recurring Template Pattern）

**用途**：LlmForCausalLMImplBase<LlmModelType>

**优点**：
- 静态多态（编译时确定，无虚函数开销）
- 代码复用（共享 forward、logits 实现）
- 类型安全

### 8.2 Traits 检测机制

**用途**：在 CausalLMImpl 中选择性调用 Model 方法

**优点**：
- 可选方法支持（无需强制实现所有方法）
- 编译时检测（高效）
- 编译期分支（if constexpr）

### 8.3 工厂模式 + 静态注册

**用途**：ModelRegistry + REGISTER_CAUSAL_MODEL

**优点**：
- 配置驱动创建
- 易于扩展新模型
- 解耦创建逻辑

---

## 9. 总结

xLLM 的 Model 层架构体现了高度的工程化设计：

1. **清晰的分层结构**：基类 → 模板包装器 → 具体实现 → Transformer → Decoder Layer
2. **灵活的 Traits 检测**：可选方法支持，编译期优化
3. **优雅的注册机制**：静态注册 + 工厂模式，易于扩展
4. **高效的设计模式**：CRTP + Pimpl，兼顾性能和可维护性

这种设计使得添加新模型只需：
1. 定义 Decoder Layer、Model、ForCausalLM 三层
2. 使用 REGISTER_CAUSAL_MODEL 注册
3. 使用 REGISTER_MODEL_ARGS 加载参数

即可自动集成到 xLLM 框架中，享受所有优化和特性。