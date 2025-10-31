/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <algorithm>

#include "core/util/utils.h"

#ifdef USE_NPU
#include <atb/atb_infer.h>
#include <mstx/ms_tools_ext.h>

#include "atb_speed/log.h"
#include "core/layers/attention_mask.h"
#include "core/layers/lm_head.h"
#include "core/layers/onerec_block_layer.h"
#include "core/layers/rms_norm.h"
#include "core/layers/word_embedding.h"
#endif

#include <glog/logging.h>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/layers/linear.h"
#include "core/util/tensor_helper.h"
#include "models/model_registry.h"

namespace xllm {

// Helper function to pad encoder output from [ntokens, hidden_size] to [bs,
// max_seq_len, hidden_size]
inline torch::Tensor pad_encoder_output(const torch::Tensor& encoder_output,
                                        const ModelInputParams& input_params) {
  const int64_t bs = input_params.rec_params->bs;
  const int64_t hidden_size = encoder_output.size(1);

  // Get actual sequence lengths and max sequence length from input_params
  const auto& seq_lens = input_params.rec_params->encoder_seq_lens;
  const int64_t max_seq_len = input_params.rec_params->encoder_max_seq_len;

  // Split encoder_output into individual sequences
  std::vector<torch::Tensor> seq_list;
  seq_list.reserve(bs);

  int64_t token_offset = 0;
  for (int64_t i = 0; i < bs; ++i) {
    const int64_t seq_len = seq_lens[i];
    seq_list.emplace_back(encoder_output.narrow(0, token_offset, seq_len));
    token_offset += seq_len;
  }

  // Use PyTorch's built-in padding function for better performance
  auto padded_output = torch::nn::utils::rnn::pad_sequence(
      seq_list, /*batch_first=*/true, /*padding_value=*/0.0);

  // Ensure the output has the correct max_seq_len dimension
  if (padded_output.size(1) < max_seq_len) {
    auto extra_padding =
        torch::zeros({bs, max_seq_len - padded_output.size(1), hidden_size},
                     encoder_output.options());
    padded_output = torch::cat({padded_output, extra_padding}, /*dim=*/1);
  }

  return padded_output;
}

#ifdef USE_NPU
class OneRecBlockImpl : public torch::nn::Module {
 public:
  OneRecBlockImpl(const ModelContext& context,
                  int layer_idx = 0,
                  bool is_decoder = true) {
    // register submodules
    block_layer_ = register_module(
        "block_layer", layer::OneRecBlockLayer(context, is_decoder, layer_idx));
  }

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        atb::Context* context,
                        AtbWorkspace& work_space,
                        std::vector<aclrtEvent*> event,
                        std::vector<std::atomic<bool>*> event_flag,
                        int layer_id,
                        const torch::Tensor& encoder_output = torch::Tensor(),
                        const torch::Tensor& expert_array = torch::Tensor()) {
    // ONEREC now passes position_bias through attn_mask with ALIBI mask type
    // Pass encoder_output to the underlying block_layer_
    return block_layer_->forward(
        x,
        attn_mask,
        kv_cache,
        input_params,
        context,
        work_space,
        event,
        event_flag,
        encoder_output.defined() ? const_cast<torch::Tensor*>(&encoder_output)
                                 : nullptr,
        layer_id,
        expert_array);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    block_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    block_layer_->verify_loaded_weights(prefix);
  }

  void merge_loaded_weights() { block_layer_->merge_loaded_weights(); }

 private:
  layer::OneRecBlockLayer block_layer_{nullptr};
};
TORCH_MODULE(OneRecBlock);

// ONEREC position bias computation similar to ONERECAttention.compute_bias
// Optimized for different stages:
// - Encoder: only prefill stage, bidirectional attention, can handle long
// sequences (~2000 tokens)
// - Decoder prefill: single token, no causal mask needed
// - Decoder decode: incremental generation, only compute bias for last query
// position
inline torch::Tensor compute_onerec_position_bias(
    int64_t query_length,
    int64_t key_length,
    int64_t num_heads,
    bool is_decoder,
    layer::WordEmbedding& position_bias_embedding,
    atb::Context* context,
    AtbWorkspace& workspace,
    int64_t num_buckets = 32,
    int64_t max_distance = 128,
    const torch::TensorOptions& options = torch::kFloat32,
    bool is_decode_stage = false,
    const ModelInputParams* input_params = nullptr) {
  auto device = options.device();
  auto dtype = options.dtype();

  // For decoder decode stage, we need full key_length but only last query
  int64_t actual_query_length = is_decode_stage ? key_length : query_length;

  // Ensure minimum valid dimensions to avoid empty tensors
  if (actual_query_length <= 0) {
    LOG(WARNING) << "[ONEREC DEBUG] actual_query_length <= 0 ("
                 << actual_query_length << "), using 1";
    actual_query_length = 1;
  }
  if (key_length <= 0) {
    LOG(WARNING) << "[ONEREC DEBUG] key_length <= 0 (" << key_length
                 << "), using 1";
    key_length = 1;
  }

  // LOG(INFO) << "[ONEREC DEBUG] compute_onerec_position_bias - query_length: "
  //           << query_length << ", key_length: " << key_length
  //           << ", actual_query_length: " << actual_query_length
  //           << ", is_decode_stage: " << is_decode_stage;

  // Create position indices
  auto context_position =
      torch::arange(actual_query_length,
                    torch::dtype(torch::kLong).device(device))
          .unsqueeze(1);
  auto memory_position =
      torch::arange(key_length, torch::dtype(torch::kLong).device(device))
          .unsqueeze(0);

  // Calculate relative position: memory_position - context_position
  auto relative_position = memory_position - context_position;

  // Convert to relative position buckets (similar to ONEREC's
  // _relative_position_bucket)
  auto relative_buckets = torch::zeros_like(relative_position);

  if (!is_decoder) {
    // Bidirectional for encoder
    num_buckets = num_buckets / 2;
    relative_buckets += (relative_position > 0).to(torch::kLong) * num_buckets;
    relative_position = torch::abs(relative_position);
  } else {
    // Unidirectional for decoder
    relative_position =
        -torch::min(relative_position, torch::zeros_like(relative_position));
  }

  // Half buckets for exact increments
  auto max_exact = num_buckets / 2;
  auto is_small = relative_position < max_exact;

  // Logarithmic buckets for larger distances
  auto relative_position_if_large =
      max_exact + (torch::log(relative_position.to(torch::kFloat) / max_exact) /
                   std::log(static_cast<double>(max_distance) / max_exact) *
                   (num_buckets - max_exact))
                      .to(torch::kLong);

  relative_position_if_large =
      torch::min(relative_position_if_large,
                 torch::full_like(relative_position_if_large, num_buckets - 1));

  relative_buckets +=
      torch::where(is_small, relative_position, relative_position_if_large);

  // Use the learned position bias embedding table
  // AtbWordEmbedding expects 1D input tensor, so we need to flatten
  // relative_buckets
  auto original_shape = relative_buckets.sizes();
  auto flattened_buckets = relative_buckets.flatten();

  auto values = position_bias_embedding(flattened_buckets, 0);

  // Handle AtbWordEmbedding output: since unpadInputs=true, it returns 2D
  // [num_tokens, hidden_size] We need to reshape it to [query_length,
  // key_length, num_heads]
  if (values.dim() == 2) {
    if (values.size(0) == flattened_buckets.size(0)) {
      // values is [flattened_size, num_heads], reshape to [query_length,
      // key_length, num_heads]
      values =
          values.view({original_shape[0], original_shape[1], values.size(1)});
      // LOG(INFO) << "[ONEREC DEBUG] Reshaped 2D values from embedding: "
      //           << values.sizes();
    } else {
      LOG(FATAL) << "[ONEREC DEBUG] Unexpected 2D values size: "
                 << values.sizes()
                 << ", expected first dim: " << flattened_buckets.size(0);
    }
  } else if (values.dim() == 1) {
    // values is [flattened_size], add num_heads dimension and reshape
    values =
        values.unsqueeze(-1).expand({flattened_buckets.size(0), num_heads});
    values = values.view({original_shape[0], original_shape[1], num_heads});
    // LOG(INFO) << "[ONEREC DEBUG] Expanded and reshaped 1D values: "
    //           << values.sizes();
  } else {
    LOG(FATAL) << "[ONEREC DEBUG] Unexpected values tensor dimension: "
               << values.dim() << ", sizes: " << values.sizes();
  }

  // Debug: Log tensor dimensions before permute
  // LOG(INFO) << "[ONEREC DEBUG] Before permute - values.sizes(): " <<
  // values.sizes()
  //           << ", relative_buckets.sizes(): " << relative_buckets.sizes()
  //           << ", query_length: " << query_length
  //           << ", key_length: " << key_length << ", num_heads: " << num_heads
  //           << ", is_decoder: " << is_decoder;

  // Now values should be [query_length, key_length, num_heads] after reshaping
  // LOG(INFO) << "[ONEREC DEBUG] After embedding reshape - values.sizes(): "
  //           << values.sizes() << ", expected: [" << actual_query_length <<
  //           ","
  //           << key_length << ", " << num_heads << "]";

  if (values.dim() == 3) {
    // values is [query_length, key_length, num_heads], permute to [num_heads,
    // query_length, key_length] ATB ALIBI mask type requires 3D tensor, not
    // 4D, so we don't add batch dimension
    values = values.permute({2, 0, 1});
    // LOG(INFO) << "[ONEREC DEBUG] 3D values after permute - values.sizes(): "
    //               << values.sizes();
    // LOG(INFO) << "position bias after permute " << values
    //               << ", value device: " << values.device();
  } else if (values.dim() == 2) {
    // Fallback: if still 2D, assume it's [query_length, key_length] and add
    // num_heads dimension
    values = values.unsqueeze(-1).expand(
        {values.size(0), values.size(1), num_heads});
    values = values.permute({2, 0, 1});
    // LOG(INFO) << "[ONEREC DEBUG] Fallback 2D handling - values.sizes(): "
    //           << values.sizes();
  } else {
    LOG(FATAL) << "[ONEREC DEBUG] Unexpected values tensor dimension: "
               << values.dim() << ", sizes: " << values.sizes();
  }

  // For decoder decode stage, handle batch with different sequence progress
  if (is_decode_stage && input_params != nullptr &&
      !input_params->kv_seq_lens_vec.empty()) {
    // In decode stage with batch processing, each sequence may have different
    // progress Use max(kv_cu_seq_lens_vec) for query_length and key_length,
    // then slice for each sequence
    /*
    int batch_size = input_params->kv_cu_seq_lens_vec.size();
    std::vector<torch::Tensor> req_bias_vec;
    req_bias_vec.reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
      // Each sequence takes its corresponding column from the position bias
      // matrix
      int seq_kv_len = input_params->kv_cu_seq_lens_vec[i];
      // Take the last query row and slice to the sequence's kv length
      // values is now 3D [num_heads, query_length, key_length]
      auto req_bias_slice =
          values.slice(1, -1, values.size(1)).slice(2, 0, seq_kv_len);
      req_bias_vec.emplace_back(req_bias_slice);
    }
    values = torch::cat(req_bias_vec, 2);  // Concatenate along key dimension
    */
    int seq_kv_len = input_params->kv_seq_lens_vec[0];
    // Take the last query row and slice to the sequence's kv length
    // values is now 3D [num_heads, query_length, key_length]
    values = values.slice(1, -1, values.size(1)).slice(2, 0, seq_kv_len);
  } else if (is_decode_stage) {
    // Original logic for single sequence or when input_params is not available
    // values is now 3D [num_heads, query_length, key_length]
    values = values.slice(1, -1, values.size(1));  // Take last query row
  }

  return values;
}

#endif

class OneRecStackImpl : public torch::nn::Module {
 public:
  OneRecStackImpl(const ModelContext& context,
                  bool is_decode,
                  layer::WordEmbedding& embed_tokens) {
#ifdef USE_NPU
    auto args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    hidden_size_ = args.hidden_size();
    // register submodules
    blocks_ = register_module("block", torch::nn::ModuleList());
    uint32_t num_layers = is_decode ? args.n_layers() : args.n_encoder_layers();
    layers_.reserve(num_layers);

    is_decoder_ = is_decode;
    use_absolute_position_embedding_ = args.use_absolute_position_embedding();
    use_moe_ = args.use_moe();
    num_experts_per_tok_ = args.num_experts_per_tok();
    relative_attention_num_buckets_ = args.relative_attention_num_buckets();
    relative_attention_max_distance_ = args.relative_attention_max_distance();
    work_space_ = AtbWorkspace(options.device());

    // share the word embedding
    embed_tokens_ = embed_tokens;
    num_heads_ = is_decode ? args.decoder_n_heads() : args.n_heads();

    // Initialize position bias embedding table for relative attention only when
    // not using absolute position embedding This replaces the random embedding
    // table in compute_onerec_position_bias
    if (!use_absolute_position_embedding_) {
      position_bias_embedding_ = register_module("position_bias_embedding",
                                                 layer::WordEmbedding(context));
    }

    norm_ = register_module("final_layer_norm", layer::RmsNorm(context));

    // Initialize rotary position embeddings (for compatibility)
    cos_pos_ = torch::Tensor();
    sin_pos_ = torch::Tensor();

    // Initialize attention mask
    int32_t mask_value = -9984;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    max_seq_len_ = args.max_position_embeddings();
    atb::Status st = atb::CreateContext(&context_);
    LOG_IF(ERROR, st != 0) << "ContextFactory create atb::Context fail";
    device_id = options.device().index();
    void* stream = c10_npu::getCurrentNPUStream(device_id).stream();
    LOG_IF(ERROR, stream == nullptr) << "get current stream fail";
    // context_->SetExecuteStream(atb_speed::Utils::GetCurrentStream());
    context_->SetExecuteStream(stream);
    context_->SetAsyncTilingCopyStatus(true);
    for (int32_t i = 0; i < num_layers; i++) {
      auto block = OneRecBlock(context, i, is_decode);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

#endif
  }

  ~OneRecStackImpl() {
    atb::Status st = atb::DestroyContext(context_);
    LOG_IF(ERROR, st != 0) << "DestroyContext atb::Context fail";
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params,
                        const torch::Tensor& encoder_output = torch::Tensor()) {
#ifdef USE_NPU
    // Get embeddings
    torch::Tensor h;
    if (input_params.rec_params->is_hybrid_mode && !is_decoder_) {
      h = tokens;
    } else {
      if (input_params.rec_params->decoder_context_embedding.defined()) {
        // use context embedding replacing bos + prompt tokens
        if (tokens.sizes() == 0) {
          h = input_params.rec_params->decoder_context_embedding;
        } else {
          h = embed_tokens_(tokens, 0);

          // Reshape tensors for interleaving
          // decoder_context_embedding: [bs * group_width * seq_len2,
          // hidden_size] h: [bs * group_width * seq_len1, hidden_size]
          auto& context_emb =
              input_params.rec_params->decoder_context_embedding;
          auto& token_emb = h;
          const int64_t hidden_size = context_emb.size(3);
          const int64_t bs = input_params.rec_params->bs;
          const int64_t group_width = input_params.rec_params->group_width;

          const int64_t context_total_tokens = context_emb.size(2);
          const int64_t token_total_tokens = token_emb.size(0);

          // Assume bs * group_width is the same for both tensors
          // We need to determine seq_len1 and seq_len2 from the tensor shapes
          // For now, assume seq_len2 is provided via input_params.seq_len or
          // can be inferred
          const int64_t bs_group = bs * group_width;
          const int64_t seq_len1 = token_total_tokens / bs_group;

          // Use pre-allocated combined space from batch.cpp
          // context_emb is already in shape [bs, group_width, total_len,
          // hidden_size] The first seq_len2 part has been filled with
          // context_embedding
          const int64_t total_len = context_total_tokens;
          const int64_t seq_len2 = total_len - seq_len1;

          // token_emb shape is [bs * group_width * seq_len1, hidden_size]
          // Need to reshape to [bs, group_width, seq_len1,
          // hidden_size] for corresponding copying
          auto token_embedding_reshaped =
              token_emb.view({bs, group_width, seq_len1, hidden_size});

          // Copy token_embedding to the last seq_len1 part of context_emb
          // Use narrow to slice from seq_len2 position in dimension 2, taking
          // seq_len1 length
          context_emb.narrow(2, seq_len2, seq_len1)
              .copy_(token_embedding_reshaped);

          // Reshape to final shape
          h = context_emb.view({-1, hidden_size}).clone();
        }
        if (!h.is_contiguous()) {
          h = h.contiguous();
        }

      } else {
        h = embed_tokens_(tokens, 0);
      }
    }

    // Ensure encoder_output is on NPU device if provided
    torch::Tensor npu_encoder_output = encoder_output;
    if (encoder_output.defined() &&
        encoder_output.device().type() != h.device().type()) {
      npu_encoder_output = encoder_output.to(h.device());
    }

    // Since unpadInputs=true in AtbWordEmbeddingImpl, h is 2D: [total_tokens,
    // hidden_size] We need to get sequence info from input_params instead auto
    // total_tokens = h.size(0); auto hidden_size = h.size(1); Get batch_size
    // and seq_length from input_params
    auto batch_size = input_params.num_sequences;
    auto seq_length = input_params.q_max_seq_len;

    // Determine stage based on input_params
    bool is_prefill = input_params.rec_params->rec_stage ==
                      RecModelInputParams::RecStage::PREFILL;

    // Compute sequence lengths for position bias calculation
    auto [query_length, key_length] =
        compute_sequence_lengths(seq_length, is_prefill, input_params);

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    bool is_decode_stage = is_decoder_ && !is_prefill;

    // Compute attention mask based on MoE usage
    torch::Tensor effective_attn_mask;
    if (use_absolute_position_embedding_) {
      effective_attn_mask =
          create_moe_attention_mask(query_length, h, is_decoder_);
    } else {
      effective_attn_mask = compute_position_bias_mask(
          query_length, key_length, h, is_decode_stage, input_params);
    }

    // Pre-process attention mask for better performance
    torch::Tensor preprocessed_attn_mask =
        preprocess_attention_mask(effective_attn_mask, h);
    torch::Tensor preprocessed_encoder_seq_lens_tensor;

    // Pre-process encoder_seq_lens_tensor if defined
    if (input_params.rec_params->encoder_seq_lens_tensor.defined()) {
      auto target_device = h.device();
      if (input_params.rec_params->encoder_seq_lens_tensor.device() !=
          target_device) {
        auto flattened_tensor =
            input_params.rec_params->encoder_seq_lens_tensor.flatten();
        preprocessed_encoder_seq_lens_tensor =
            flattened_tensor.to(target_device, torch::kInt).contiguous();
      } else {
        auto flattened_tensor =
            input_params.rec_params->encoder_seq_lens_tensor.flatten().to(
                torch::kInt);
        preprocessed_encoder_seq_lens_tensor =
            flattened_tensor.is_contiguous() ? flattened_tensor
                                             : flattened_tensor.contiguous();
      }
      // Update input_params to use preprocessed tensor
      input_params_new.rec_params->encoder_seq_lens_tensor =
          preprocessed_encoder_seq_lens_tensor;
    } else {
      // Even if not defined, copy the original tensor to input_params_new
      input_params_new.rec_params->encoder_seq_lens_tensor =
          input_params.rec_params->encoder_seq_lens_tensor;
    }

    // Create expert_array tensor for MoE support
    torch::Tensor expert_array;
    if (use_moe_) {
      int64_t input_length = h.size(0);
      expert_array = torch::arange(
          0,
          input_length * num_experts_per_tok_,
          torch::TensorOptions().dtype(torch::kInt32).device(h.device()));
    }

    for (size_t i = 0; i < layers_.size(); i++) {
      if (input_params.layer_synchronizer) {
        input_params.layer_synchronizer->synchronize_layer(i);
      }

      //@TODO: init
      std::vector<aclrtEvent*> events;
      std::vector<std::atomic<bool>*> event_flags;

      auto& layer = layers_[i];
      // Use reference to kv_caches[i] to match KVCache& parameter type
      KVCache& kv_cache_ref = kv_caches[i];
      layers_[i]->forward(
          h,
          cos_pos_,
          sin_pos_,
          effective_attn_mask,  // Pass position_bias as attn_mask
          kv_cache_ref,
          input_params_new,
          context_,
          work_space_,
          events,
          event_flags,
          i,
          npu_encoder_output,
          expert_array);
    }
    h = norm_(h, 0);
    return h;

#endif
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    // Load position bias embedding weights (from first layer's relative
    // attention bias)
    if (!use_absolute_position_embedding_) {
      position_bias_embedding_->load_state_dict(state_dict.get_dict_with_prefix(
          "block.0.layer.0.SelfAttention.relative_attention_bias."));
    }
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("block." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(
        state_dict.get_dict_with_prefix("final_layer_norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    if (!use_absolute_position_embedding_) {
      position_bias_embedding_->verify_loaded_weights(
          prefix + "block.0.layer.0.SelfAttention.relative_attention_bias.");
    }
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "block." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "final_layer_norm.");
  }

#ifdef USE_NPU
  void merge_loaded_weights() {
    // test
    embed_tokens_->merge_loaded_weights();
    if (!use_absolute_position_embedding_) {
      position_bias_embedding_->merge_loaded_weights();
    }
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }
#endif

 private:
  int64_t hidden_size_;

#ifdef USE_CUDA
  // parameter members, must be registered
  ParallelEmbedding embed_tokens_{nullptr};
  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};
  layer::RMSNorm norm_{nullptr};
#endif
#ifdef USE_NPU
  torch::Tensor cos_pos_;
  torch::Tensor sin_pos_;
  torch::Tensor position_bias_;
  atb::Context* context_;
  int max_seq_len_ = 0;
  int device_id = 0;
  bool is_decoder_;
  bool use_absolute_position_embedding_ = false;
  bool use_moe_ = false;
  int64_t relative_attention_num_buckets_ = 32;
  int64_t relative_attention_max_distance_ = 128;
  int64_t num_heads_ = 4;
  int32_t num_experts_per_tok_ = 2;
  AtbWorkspace work_space_;
  layer::AttentionMask attn_mask_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::WordEmbedding position_bias_embedding_{nullptr};
  layer::RmsNorm norm_{nullptr};
#endif

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<OneRecBlock> layers_;

  // Helper functions for position bias computation
  std::pair<int64_t, int64_t> compute_sequence_lengths(
      int64_t seq_length,
      bool is_prefill,
      const ModelInputParams& input_params) const;
  torch::Tensor create_moe_attention_mask(int64_t seq_length,
                                          const torch::Tensor& h,
                                          bool is_decoder) const;
  torch::Tensor compute_position_bias_mask(
      int64_t query_length,
      int64_t key_length,
      const torch::Tensor& h,
      bool is_decode_stage,
      const ModelInputParams& input_params);
  torch::Tensor preprocess_attention_mask(
      const torch::Tensor& effective_attn_mask,
      const torch::Tensor& h) const;
};
TORCH_MODULE(OneRecStack);

class OneRecForConditionalGenerationImpl : public torch::nn::Module {
 public:
  OneRecForConditionalGenerationImpl(const ModelContext& context) {
#ifdef USE_NPU
    auto args = context.get_model_args();
    auto options = context.get_tensor_options();

    device_id = options.device().index();
    work_space_ = AtbWorkspace(options.device());
    use_moe_ = args.use_moe();

    shared_ = register_module("shared", layer::WordEmbedding(context));

    // Only initialize encoder when use_moe is false
    bool is_decode = false;
    encoder_ =
        register_module("encoder", OneRecStack(context, is_decode, shared_));

    is_decode = true;
    decoder_ =
        register_module("decoder", OneRecStack(context, is_decode, shared_));

    lm_head_ = register_module("lm_head", layer::LmHead(context));

    atb::Status st = atb::CreateContext(&context_);
    LOG_IF(ERROR, st != 0) << "ContextFactory create atb::Context fail";

    void* stream = c10_npu::getCurrentNPUStream(device_id).stream();
    LOG_IF(ERROR, stream == nullptr) << "get current stream fail";
    tie_word_embeddings_ = args.tie_word_embeddings();
    scale_factor_ = 1 / sqrt(args.hidden_size());
    context_->SetExecuteStream(stream);
    context_->SetAsyncTilingCopyStatus(true);

#endif
  }

  ~OneRecForConditionalGenerationImpl() {
    atb::Status st = atb::DestroyContext(context_);
    LOG_IF(ERROR, st != 0) << "DestroyContext atb::Context fail";
  }
  // Encoder forward pass - processes encoder input tokens
  // encoder_tokens: [num_encoder_tokens] encoder input tokens
  // encoder_positions: [num_encoder_tokens] encoder token positions
  // returns: [num_encoder_tokens, hidden_size] encoder hidden states
  torch::Tensor encode_forward(const torch::Tensor& encoder_tokens,
                               const torch::Tensor& encoder_positions,
                               const ModelInputParams& input_params) {
    // Run encoder with encoder input tokens
    std::vector<KVCache> encoder_kv_caches;  // Encoder doesn't use KV cache

    auto encoder_output = encoder_(
        encoder_tokens, encoder_positions, encoder_kv_caches, input_params);
    encoder_output = pad_encoder_output(encoder_output, input_params);
    encoder_output_ = encoder_output;
    return encoder_output;
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params,
                        const torch::Tensor& encoder_output = torch::Tensor()) {
    // ONEREC decoder forward pass with cross-attention to encoder output
    auto decoder_output =
        decoder_(tokens, positions, kv_caches, input_params, encoder_output_);
    return decoder_output;
  }

  torch::Tensor forward(std::vector<torch::Tensor> tokens,
                        std::vector<torch::Tensor> positions,
                        std::vector<KVCache>& kv_caches,
                        const std::vector<ModelInputParams>& input_params) {
    if (input_params[0].rec_params->is_encoder_forward) {
      return encode_forward(tokens[0], positions[0], input_params[0]);
    }
    return forward(
        tokens[0], positions[0], kv_caches, input_params[0], encoder_output_);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    // select tokens if provided
    auto h = hidden_states;
    if (tie_word_embeddings_) {
      h = hidden_states * scale_factor_;
    }
#ifdef USE_NPU
    return lm_head_(h, seleted_idxes, 0);

#endif
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }

  virtual void update_expert_weight(int32_t layer_id) { return; }

  // TODO load model
  void load_model(std::unique_ptr<ModelLoader> loader) {
#ifdef USE_NPU
    for (const auto& state_dict : loader->get_state_dicts()) {
      shared_->load_state_dict(state_dict->get_dict_with_prefix("shared."));
      encoder_->load_state_dict(state_dict->get_dict_with_prefix("encoder."));
      decoder_->load_state_dict(state_dict->get_dict_with_prefix("decoder."));
      if (tie_word_embeddings_) {
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("shared."));
      } else {
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
      }
    }
    // verify
    shared_->verify_loaded_weights("shared.");
    encoder_->verify_loaded_weights("encoder.");
    decoder_->verify_loaded_weights("decoder.");
    lm_head_->verify_loaded_weights("lm_head.");

    shared_->merge_loaded_weights();
    encoder_->merge_loaded_weights();
    decoder_->merge_loaded_weights();
    lm_head_->merge_loaded_weights();
    LOG(INFO) << "load model done";
#endif
  }

#ifdef USE_NPU
  layer::LmHead get_lm_head() { return lm_head_; }

  void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  std::vector<layer::WordEmbedding> get_word_embedding() { return {shared_}; }

  void set_word_embedding(std::vector<layer::WordEmbedding>& embedding) {
    shared_ = embedding[0];
  }
#endif

 private:
#ifdef USE_NPU
  float scale_factor_;
  bool tie_word_embeddings_{false};
  bool use_moe_ = false;
#endif
  int device_id = 0;
  layer::WordEmbedding shared_{nullptr};
  OneRecStack encoder_{nullptr};
  OneRecStack decoder_{nullptr};
  layer::LmHead lm_head_{nullptr};
  AtbWorkspace work_space_;
  atb::Context* context_;
  torch::Tensor
      encoder_output_;  // Store encoder output for decoder cross-attention
#ifndef USE_NPU
  ColumnParallelLinear lm_head_{nullptr};
#endif
};
TORCH_MODULE(OneRecForConditionalGeneration);

#ifdef USE_NPU
// Implementation of OneRecStackImpl helper functions
inline std::pair<int64_t, int64_t> OneRecStackImpl::compute_sequence_lengths(
    int64_t seq_length,
    bool is_prefill,
    const ModelInputParams& input_params) const {
  int64_t query_length = seq_length;
  int64_t key_length = seq_length;

  if (is_decoder_) {
    // ONEREC Decoder logic
    if (is_prefill) {
      // Decoder prefill: query_length = decoder input length, key_length =
      // decoder input length for self-attn
      query_length = seq_length;
      key_length = seq_length;
    } else {
      // Decoder decode: query_length = 1 (new token), key_length = accumulated
      // length
      query_length = 1;
      if (!input_params.kv_seq_lens_vec.empty()) {
        auto max_kv_len =
            *std::max_element(input_params.kv_seq_lens_vec.begin(),
                              input_params.kv_seq_lens_vec.end());
        key_length = max_kv_len;
      } else {
        key_length = seq_length;
      }
      // For position bias slicing in different sequences
      query_length = key_length;
    }
  } else {
    // ONEREC Encoder logic: always prefill stage with full input sequence
    // Use bidirectional attention for full input sequence
    // Use encoder_max_seq_len instead of q_max_seq_len for correct position
    // bias calculation
    auto encoder_seq_length = input_params.rec_params->encoder_max_seq_len;
    query_length = encoder_seq_length;
    key_length = encoder_seq_length;
  }

  return {query_length, key_length};
}

inline torch::Tensor OneRecStackImpl::create_moe_attention_mask(
    int64_t seq_length,
    const torch::Tensor& h,
    bool is_decoder) const {
  // When use_moe is true, skip position bias computation and use triangular
  // mask directly
  if (!is_decoder) {
    auto effective_attn_mask =
        torch::ones({num_heads_, seq_length, seq_length}, h.options());
    return effective_attn_mask;
  }
  auto mask_value = -9984.0f;
  // Create upper triangular mask (offset=1 to exclude diagonal)
  auto upper_tri_mask =
      torch::triu(torch::ones({seq_length, seq_length},
                              torch::dtype(h.dtype()).device(h.device())),
                  1);
  // Expand mask to match dimensions [num_heads, seq_len, seq_len]
  auto expanded_mask =
      upper_tri_mask.unsqueeze(0).expand({num_heads_, seq_length, seq_length});

  // Create base mask filled with zeros
  auto effective_attn_mask =
      torch::zeros({num_heads_, seq_length, seq_length},
                   torch::dtype(h.dtype()).device(h.device()));
  // Apply triangular mask
  effective_attn_mask.masked_fill_(expanded_mask.to(torch::kBool), mask_value);
  return effective_attn_mask;
}

inline torch::Tensor OneRecStackImpl::compute_position_bias_mask(
    int64_t query_length,
    int64_t key_length,
    const torch::Tensor& h,
    bool is_decode_stage,
    const ModelInputParams& input_params) {
  // Compute position bias for the first layer
  auto layer_position_bias =
      compute_onerec_position_bias(query_length,
                                   key_length,
                                   num_heads_,
                                   is_decoder_,
                                   position_bias_embedding_,
                                   context_,
                                   work_space_,
                                   relative_attention_num_buckets_,
                                   relative_attention_max_distance_,
                                   torch::dtype(h.dtype()).device(h.device()),
                                   is_decode_stage,
                                   &input_params);

  // Generate appropriate attention mask based on encoder/decoder type
  auto effective_attn_mask = layer_position_bias.is_contiguous()
                                 ? layer_position_bias
                                 : layer_position_bias.contiguous();

  if (is_decoder_ && FLAGS_enable_rec_prefill_only) {
    // Use torch::triu to create upper triangular mask and apply it
    auto mask_value = -9984.0f;
    // Create upper triangular mask (offset=1 to exclude diagonal)
    auto upper_tri_mask =
        torch::triu(torch::ones({query_length, query_length},
                                effective_attn_mask.options()),
                    1);
    // Expand mask to match effective_attn_mask dimensions [num_heads, seq_len,
    // seq_len]
    auto expanded_mask = upper_tri_mask.unsqueeze(0).expand(
        {num_heads_, query_length, query_length});

    // Apply mask to all heads using broadcasting (single operation)
    effective_attn_mask.masked_fill_(expanded_mask.to(torch::kBool),
                                     mask_value);
  }

  return effective_attn_mask;
}

inline torch::Tensor OneRecStackImpl::preprocess_attention_mask(
    const torch::Tensor& effective_attn_mask,
    const torch::Tensor& h) const {
  if (!effective_attn_mask.defined()) {
    return torch::Tensor();
  }

  // Check device compatibility
  auto target_device = h.device();
  if (effective_attn_mask.device() != target_device) {
    LOG(WARNING) << "[ONEREC Optimization] Moving attn_mask from device "
                 << effective_attn_mask.device() << " to " << target_device;
    return effective_attn_mask.to(target_device).contiguous();
  } else {
    return effective_attn_mask.is_contiguous()
               ? effective_attn_mask
               : effective_attn_mask.contiguous();
  }
}
#endif

// register the causal model
REGISTER_CAUSAL_MODEL(onerec, OneRecForConditionalGeneration);

// register the model args
REGISTER_MODEL_ARGS(onerec, [&] {
  LOAD_ARG_OR(model_type, "model_type", "onerec");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG(decoder_n_kv_heads, "decoder_num_key_value_heads");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_heads, "num_heads", 4);
  LOAD_ARG_OR(head_dim, "d_kv", 4);
  LOAD_ARG_OR_FUNC(
      decoder_n_heads, "decoder_num_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR_FUNC(
      decoder_head_dim, "decoder_d_kv", [&] { return args->head_dim(); });
  // decide model type based on vocab size
  LOAD_ARG_OR(vocab_size, "vocab_size", 8200);
  LOAD_ARG_OR(n_layers, "num_decoder_layers", 4);
  LOAD_ARG_OR(n_encoder_layers, "num_layers", 12);
  LOAD_ARG_OR(rms_norm_eps, "layer_norm_epsilon", 1e-6);
  LOAD_ARG_OR(max_position_embeddings, "max_length", 500);
  LOAD_ARG_OR(intermediate_size, "d_ff", 256);
  LOAD_ARG_OR(hidden_size, "d_model", 128);
  LOAD_ARG_OR(use_absolute_position_embedding,
              "use_absolute_position_embedding",
              false);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", true);
  // moe. reuse deepseekv2
  LOAD_ARG_OR(use_moe, "use_moe", false);
  LOAD_ARG_OR(moe_score_func, "moe_score_func", "softmax");
  LOAD_ARG_OR(moe_route_scale, "moe_route_scale", 1.0);
  LOAD_ARG_OR(n_routed_experts, "moe_num_experts", 8);
  LOAD_ARG_OR(moe_use_shared_experts, "moe_use_shared_experts", false);
  LOAD_ARG_OR(n_shared_experts, "moe_num_shared_experts", 0);
  LOAD_ARG_OR(num_experts_per_tok, "moe_topk", 2);
  LOAD_ARG_OR(moe_intermediate_size, "moe_inter_dim", 1024);

  LOAD_ARG_OR(
      relative_attention_num_buckets, "relative_attention_num_buckets", 32);
  LOAD_ARG_OR(
      relative_attention_max_distance, "relative_attention_max_distance", 128);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 128001);

  // LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
  //   return args->hidden_size() / args->n_heads();
  // });
});

REGISTER_TOKENIZER_ARGS(onerec, [&] { SET_ARG(tokenizer_type, "rec"); });

}  // namespace xllm