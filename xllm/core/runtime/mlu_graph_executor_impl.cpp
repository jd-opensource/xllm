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

#include "mlu_graph_executor_impl.h"

#include <cnrt.h>
#include <torch_mlu/csrc/framework/core/stream_guard.h>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "util/utils.h"

namespace {
// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t get_bucket_num_tokens(uint32_t num_tokens) {
  if (FLAGS_enable_graph_no_padding) {
    return num_tokens;
  }
  const uint32_t graph_step = 16;
  if (num_tokens <= 1) return 1;
  if (num_tokens <= 2) return 2;
  if (num_tokens <= 4) return 4;
  if (num_tokens <= 8) return 8;

  return ((num_tokens + graph_step - 1) / graph_step) * graph_step;
}
}  // namespace

namespace xllm {

GraphPersistentParam::GraphPersistentParam(const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : num_decoding_tokens_(options.num_decoding_tokens()) {
  const int64_t max_tokens = FLAGS_max_tokens_per_batch;
  const int64_t max_seqs = options.max_seqs_per_batch();
  const int64_t max_seq_len = FLAGS_max_seq_len_for_graph_mode > 0
                                  ? FLAGS_max_seq_len_for_graph_mode
                                  : args.max_position_embeddings();
  const uint32_t block_size = options.block_size();
  const int64_t max_num_blocks_per_req =
      (max_seq_len + block_size - 1) / block_size + 1;
  torch::ScalarType torch_type = util::parse_dtype(args.dtype(), device);
  auto tensor_options = torch::TensorOptions().device(device).dtype(torch_type);
  auto int_tensor_options = tensor_options.dtype(torch::kInt32);

  // output buffer
  output_ = torch::zeros({max_tokens, args.hidden_size()}, tensor_options);

  // input buffers
  if (args.rope_scaling_mrope_section().empty()) {
    positions_ = torch::zeros({max_tokens}, int_tensor_options);
  } else {
    positions_ = torch::zeros({3, max_tokens}, int_tensor_options);
    use_mrope_ = true;
  }
  tokens_ = torch::zeros({max_tokens}, int_tensor_options);
  new_cache_slots_ = torch::zeros({max_tokens}, int_tensor_options);
  block_table_ =
      torch::zeros({max_tokens, max_num_blocks_per_req}, int_tensor_options);
  // Sequence length tensors with max_seqs
  q_seq_lens_ = torch::zeros({max_seqs + 1}, int_tensor_options);
  kv_seq_lens_ = torch::zeros({max_seqs + 1}, int_tensor_options);
}

void GraphPersistentParam::init_params(const ModelInputParams& params,
                                       uint32_t padding_num_tokens,
                                       uint32_t padding_needed) {
  params_ = params.to(tokens_.device());
  params_.q_seq_lens =
      q_seq_lens_.slice(0, 0, params.q_seq_lens.size(0) + padding_needed);
  params_.kv_seq_lens =
      kv_seq_lens_.slice(0, 0, params.kv_seq_lens.size(0) + padding_needed);
  params_.new_cache_slots = new_cache_slots_.slice(0, 0, padding_num_tokens);
  params_.block_tables = block_table_.slice(0, 0, padding_num_tokens);
  params_.dp_global_token_nums = std::vector<int32_t>(
      params.dp_global_token_nums.size(), padding_num_tokens);

  if (params.input_embedding.defined()) {
    if (!input_embeds_.defined()) {
      input_embeds_ = torch::zeros_like(output_);
    }
    params_.input_embedding = input_embeds_.slice(0, 0, padding_num_tokens);
  }
}

void GraphPersistentParam::update_input_buffer(const torch::Tensor& tokens,
                                               const torch::Tensor& positions,
                                               const ModelInputParams& params,
                                               uint32_t padding_needed) {
  // Copy data from input parameters to persistent graph tensors
  int32_t slice_dim = use_mrope_ ? 1 : 0;
  positions_.slice(slice_dim, 0, positions.size(slice_dim))
      .copy_(positions, true);
  tokens_.slice(0, 0, tokens.size(0)).copy_(tokens, true);
  new_cache_slots_.slice(0, 0, params.new_cache_slots.size(0))
      .copy_(params.new_cache_slots, true);

  // Apply padding if required number of tokens exceeds actual input
  // Generate padded sequence lengths by extending the last valid value
  std::vector<int32_t> q_seq_lens_vec(params.q_seq_lens_vec);
  std::vector<int32_t> kv_seq_lens_vec(params.kv_seq_lens_vec);
  if (padding_needed > 0) {
    q_seq_lens_vec.reserve(q_seq_lens_vec.size() + padding_needed);
    kv_seq_lens_vec.reserve(kv_seq_lens_vec.size() + padding_needed);
    for (size_t i = 0; i < padding_needed; i++) {
      q_seq_lens_vec.push_back(q_seq_lens_vec.back() + num_decoding_tokens_);
      kv_seq_lens_vec.push_back(kv_seq_lens_vec.back() + num_decoding_tokens_);
    }
  }
  auto q_seq_lens = torch::tensor(q_seq_lens_vec, q_seq_lens_.options());
  auto kv_seq_lens = torch::tensor(kv_seq_lens_vec, kv_seq_lens_.options());
  q_seq_lens_.slice(0, 0, q_seq_lens.size(0)).copy_(q_seq_lens, true);
  kv_seq_lens_.slice(0, 0, kv_seq_lens.size(0)).copy_(kv_seq_lens, true);

  // Copy block table data
  const int64_t actual_batch = params.block_tables.size(0);
  const int64_t actual_n_block = params.block_tables.size(1);
  auto slice_block_tables =
      block_table_.slice(0, 0, actual_batch).slice(1, 0, actual_n_block);
  slice_block_tables.copy_(params.block_tables, true);

  if (params.input_embedding.defined()) {
    input_embeds_.slice(0, 0, params.input_embedding.size(0))
        .copy_(params.input_embedding, true);
  }
}

MluGraph::MluGraph(GraphPersistentParam* persistent_param,
                   uint32_t padding_num_tokens)
    : persistent_param_(persistent_param),
      padding_num_tokens_(padding_num_tokens) {}

void MluGraph::capture(CausalLM* model,
                       std::vector<KVCache>& kv_cache,
                       torch_mlu::MempoolId_t& pool) {
  int32_t slice_dim = persistent_param_->use_mrope_ ? 1 : 0;
  torch_mlu::synchronize();
  auto prev_stream = torch_mlu::getCurrentMLUStream();
  torch_mlu::mlu::MLUStreamGuard guard(torch_mlu::getStreamFromPool());
  graph_ = torch_mlu::MLUGraph();
  graph_.capture_begin(pool, cnrtQueueCaptureModeRelaxed);
  auto forward_result = model->forward(
      persistent_param_->tokens_.slice(0, 0, padding_num_tokens_),
      persistent_param_->positions_.slice(slice_dim, 0, padding_num_tokens_),
      kv_cache,
      persistent_param_->params_);
  persistent_param_->output_.slice(0, 0, forward_result.size(0))
      .copy_(forward_result, true);
  graph_.capture_end();
  torch_mlu::setCurrentMLUStream(prev_stream);
  torch_mlu::synchronize();
  graph_.replay();
  pool = graph_.pool();
}

void MluGraph::replay() { graph_.replay(); }

void MluGraph::update_input_buffer(const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   const ModelInputParams& params,
                                   bool is_init) {
  uint32_t padding_needed = padding_num_tokens_ - tokens.size(0);
  if (is_init) {
    persistent_param_->init_params(params, padding_num_tokens_, padding_needed);
  }
  persistent_param_->update_input_buffer(
      tokens, positions, params, padding_needed);
}

MluGraphExecutorImpl::MluGraphExecutorImpl(CausalLM* model,
                                           const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : model_(model),
      args_(args),
      device_(device),
      options_(options),
      pool_(torch_mlu::MempoolId_t{0, 0}) {
  persistent_param_ =
      std::make_unique<GraphPersistentParam>(args_, device_, options_);
}

ForwardInput MluGraphExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);
}

// Main execution method with graph optimization for decode phase
// tokens: [num_decode_tokens]
// positions: [num_decode_tokens] token pos in the sequence
// returns: [num_decode_tokens, hidden_size]
torch::Tensor MluGraphExecutorImpl::run(const torch::Tensor& tokens,
                                        const torch::Tensor& positions,
                                        std::vector<KVCache>& kv_caches,
                                        const ModelInputParams& params) {
  // If not in decode phase, use eager mode directly
  bool graph_mode = params.batch_forward_type.is_decode();
  int64_t actual_num_tokens = tokens.size(0);
  if (params.dp_global_token_nums.size() > 1) {
    actual_num_tokens = util::max(params.dp_global_token_nums);

    auto& dp_is_decode = params.dp_is_decode;
    graph_mode = std::find(dp_is_decode.begin(), dp_is_decode.end(), 0) ==
                 dp_is_decode.end();
    CHECK_EQ(dp_is_decode.size(), params.dp_global_token_nums.size());
  }

  if (!graph_mode) {
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  uint32_t padding_batch_size = get_bucket_num_tokens(actual_num_tokens);
  if (auto it = graphs_.find(padding_batch_size); it != graphs_.end()) {
    MluGraph* cur_graph = (it->second).get();
    cur_graph->update_input_buffer(tokens, positions, params);
    cur_graph->replay();
  } else {
    std::unique_ptr<MluGraph> graph =
        std::make_unique<MluGraph>(persistent_param_.get(), padding_batch_size);
    graph->update_input_buffer(tokens, positions, params, true);
    graph->capture(model_, kv_caches, pool_);
    graphs_[padding_batch_size] = std::move(graph);
  }
  return persistent_param_->output_.slice(0, 0, tokens.size(0));
}

}  // namespace xllm
