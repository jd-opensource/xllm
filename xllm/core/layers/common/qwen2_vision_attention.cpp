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

#include "qwen2_vision_attention.h"

#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
namespace xllm {
namespace layer {

Qwen2VisionAttentionImpl::Qwen2VisionAttentionImpl(
    const ModelContext& context) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  const int64_t hidden_size = args.mm_hidden_size();
  const int64_t num_heads = args.mm_num_attention_heads();
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  CHECK(num_heads % tp_size == 0);

  tp_group_ = parallel_args.tp_group_;
  hidden_size_per_attention_head_ = args.mm_head_dim();
  num_attention_heads_per_partition_ = num_heads / tp_size;
  scale_ = 1.0 / std::sqrt(static_cast<float>(hidden_size_per_attention_head_));

  qkv_proj_ =
      register_module("qkv_proj",
                      QKVParallelLinear(hidden_size,
                                        num_attention_heads_per_partition_,
                                        num_attention_heads_per_partition_,
                                        hidden_size_per_attention_head_,
                                        /*num_kv_head_replicas=*/1,
                                        /*bias=*/true,
                                        /*gather_output=*/false,
                                        parallel_args,
                                        options));

  proj_ = register_module("proj",
                          RowParallelLinear(hidden_size,
                                            hidden_size,
                                            /*bias=*/true,
                                            /*input_is_parallelized=*/true,
                                            /*if_reduce_results=*/true,
                                            quant_args,
                                            parallel_args.tp_group_,
                                            options));
}

std::vector<torch::Tensor> Qwen2VisionAttentionImpl::split_qkv(
    const torch::Tensor& qkv) {
  // [s, b, 3 * head * head_dim]
  auto sizes = qkv.sizes();
  int64_t seq_len = qkv.size(0);
  int64_t bs = qkv.sizes() == 3 ? qkv.size(1) : 1;
  torch::Tensor qkv_gathered =
      xllm::parallel_state::all_gather_interleaved(qkv, tp_group_);

  // [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
  auto qkv_chunks = qkv_gathered.chunk(3, /*dim=*/-1);
  auto q = qkv_chunks[0];
  auto k = qkv_chunks[1];
  auto v = qkv_chunks[2];

  // 3 * [s, b, head * head_dim]
  if (tp_group_->world_size() > 1) {
    q = xllm::parallel_state::scatter(q, tp_group_);
    k = xllm::parallel_state::scatter(k, tp_group_);
    v = xllm::parallel_state::scatter(v, tp_group_);
  }

  // 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
  std::vector<int64_t> new_shape = {seq_len,
                                    bs,
                                    num_attention_heads_per_partition_,
                                    hidden_size_per_attention_head_};
  q = q.reshape(new_shape);
  k = k.reshape(new_shape);
  v = v.reshape(new_shape);

  return {q, k, v};
}

torch::Tensor Qwen2VisionAttentionImpl::forward(
    torch::Tensor& hidden_states,
    torch::Tensor& m_cos_pos,
    torch::Tensor& m_sin_pos,
    torch::Tensor& cu_seq_len,
    std::vector<int32_t>& cu_seq_len_vec,
    ModelInputParams& params) {
  // 1. qkv projection
  auto qkv = qkv_proj_->forward(hidden_states);
  // 2. split qkv
  auto qkv_split = split_qkv(qkv);
  // 3. transpose [s, b, h, d] -> [b, s, h, d]
  for (auto& tensor : qkv_split) {
    tensor = tensor.transpose(0, 1).contiguous();
  }
  auto q = qkv_split[0];
  auto k = qkv_split[1];
  auto v = qkv_split[2];
  int64_t B = q.size(0);
  int64_t S = q.size(1);
  int64_t head_dim = q.size(3);
  CHECK_EQ(head_dim, hidden_size_per_attention_head_) << "head_dim mismatch";

  // 4. rope
  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = q;
  rotary_params.sin = m_sin_pos;
  rotary_params.cos = m_cos_pos;
  rotary_params.interleaved = false;
  rotary_params.discrete = false;
  rotary_params.max_query_len = S;
  xllm::kernel::apply_rotary(rotary_params);
  rotary_params.q = k;
  xllm::kernel::apply_rotary(rotary_params);

  // q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
  q = q.view({B * S, q.size(2), q.size(3)});
  k = k.view({B * S, k.size(2), k.size(3)});
  v = v.view({B * S, v.size(2), v.size(3)});
  torch::Tensor output = torch::zeros_like(q);

  // 5. store k/v cache and do attention
  int32_t max_seqlen =
      *std::max_element(cu_seq_len_vec.begin(), cu_seq_len_vec.end());
  xllm::kernel::AttentionParams attention_params;
  attention_params.query = q;
  attention_params.key = k;
  attention_params.value = v;
  attention_params.output = output;

  attention_params.window_size_left = -1;
  attention_params.scale = scale_;
  attention_params.compute_dtype = "half";
  attention_params.q_cu_seq_lens = cu_seq_len;
  attention_params.kv_cu_seq_lens = cu_seq_len;
  attention_params.max_query_len = max_seqlen;
  attention_params.max_seq_len = max_seqlen;
  attention_params.block_table = std::nullopt;
  attention_params.is_causal = false;
  xllm::kernel::batch_prefill(attention_params);

  // context_layer = rearrange(output, "(b s) h d -> s b (h d)", b=batch_size)
  output = output.view({B, S, -1});
  // [B, S, ...] -> [S, B, ...]
  output = output.transpose(0, 1).reshape({-1, output.size(-1)});
  // 6. output projection
  return proj_->forward(output);
}

void Qwen2VisionAttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict.get_dict_with_prefix("qkv."));
  proj_->load_state_dict(state_dict.get_dict_with_prefix("proj."));
}

}  // namespace layer
}  // namespace xllm
