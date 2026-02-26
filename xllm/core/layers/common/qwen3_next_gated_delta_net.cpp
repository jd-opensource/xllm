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

#include "qwen3_next_gated_delta_net.h"

#include <glog/logging.h>
#include "xllm/core/kernels/ops_api.h"

#include <torch_npu/torch_npu.h>
#include <acl/acl.h>

#include <tuple>
#include <unordered_map>
#include <torch/torch.h>

namespace xllm {
namespace layer {


namespace {
torch::Tensor l2norm(const torch::Tensor& x, int64_t dim, double eps = 1e-6) {
    auto norm = torch::sqrt(torch::sum(torch::square(x), dim, true) + eps);
    return x / norm;
}

std::tuple<torch::Tensor, torch::Tensor> torch_recurrent_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    std::optional<torch::Tensor> initial_state,
    bool output_final_state = true,
    bool use_qk_l2norm_in_kernel = true
) {
    auto initial_dtype = query.dtype();

    if (use_qk_l2norm_in_kernel) {
        query = l2norm(query, -1, 1e-6);
        key = l2norm(key, -1, 1e-6);
    }

    auto to_float32_and_transpose = [](torch::Tensor x) {
        return x.transpose(1, 2).contiguous().to(torch::kFloat32);
    };
    query = to_float32_and_transpose(query);
    key = to_float32_and_transpose(key);
    value = to_float32_and_transpose(value);
    beta = to_float32_and_transpose(beta);
    g = to_float32_and_transpose(g);

    int64_t batch_size = key.size(0);
    int64_t num_heads = key.size(1);
    int64_t sequence_length = key.size(2);
    int64_t k_head_dim = key.size(3);
    int64_t v_head_dim = value.size(3);

    float scale_val = 1.0 / std::sqrt(static_cast<float>(query.size(-1)));
    torch::Tensor scale = torch::tensor(scale_val, query.options());
    query = query * scale;
    torch::Tensor core_attn_out = torch::zeros({batch_size, num_heads, sequence_length, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
    torch::Tensor last_recurrent_state;
    if (!initial_state.has_value()) {
        last_recurrent_state = torch::zeros({batch_size, num_heads, k_head_dim, v_head_dim},
            torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
    } else {
        last_recurrent_state = initial_state.value().to(value.device(), torch::kFloat32);
    }

    for (int64_t i = 0; i < sequence_length; ++i) {
        torch::Tensor q_t = query.select(2, i);
        torch::Tensor k_t = key.select(2, i);
        torch::Tensor v_t = value.select(2, i);
        torch::Tensor g_t = g.select(2, i).exp().unsqueeze(-1).unsqueeze(-1);
        torch::Tensor beta_t = beta.select(2, i).unsqueeze(-1);
        last_recurrent_state = last_recurrent_state * g_t;
        torch::Tensor kv_mem = torch::sum(last_recurrent_state * k_t.unsqueeze(-1), -2);
        torch::Tensor delta = (v_t - kv_mem) * beta_t;
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
        core_attn_out.select(2, i) = torch::sum(last_recurrent_state * q_t.unsqueeze(-1), -2);
    }

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
    return std::make_tuple(core_attn_out, last_recurrent_state);
}

std::tuple<torch::Tensor, torch::Tensor> torch_chunk_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    int64_t chunk_size = 64,
    c10::optional<torch::Tensor> initial_state = c10::nullopt,
    bool output_final_state = true,
    bool use_qk_l2norm_in_kernel = true) {
    auto initial_dtype = query.dtype();
    if (use_qk_l2norm_in_kernel) {
        query = l2norm(query, -1, 1e-6);
        key = l2norm(key, -1, 1e-6);
    }
    auto to_float32 = [](torch::Tensor x) {
        return x.transpose(1, 2).contiguous().to(torch::kFloat32);
    };

    query = to_float32(query);
    key = to_float32(key);
    value = to_float32(value);
    beta = to_float32(beta);
    g = to_float32(g);

    auto batch_size = query.size(0);
    auto num_heads = query.size(1);
    auto sequence_length = query.size(2);
    auto k_head_dim = key.size(-1);
    auto v_head_dim = value.size(-1);

    int64_t pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
    query = torch::nn::functional::pad(query, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    key = torch::nn::functional::pad(key, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    value = torch::nn::functional::pad(value, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    beta = torch::nn::functional::pad(beta, torch::nn::functional::PadFuncOptions({0, pad_size}));
    g = torch::nn::functional::pad(g, torch::nn::functional::PadFuncOptions({0, pad_size}));

    int64_t total_sequence_length = sequence_length + pad_size;
    float scale = 1.0 / std::sqrt(static_cast<float>(query.size(-1)));
    query = query * scale;
    auto v_beta = value * beta.unsqueeze(-1);
    auto k_beta = key * beta.unsqueeze(-1);
    auto reshape_to_chunks = [chunk_size](torch::Tensor x) {
        auto shape = x.sizes();
        std::vector<int64_t> new_shape = {
            shape[0], shape[1], 
            shape[2] / chunk_size, chunk_size, 
            shape[3]
        };
        return x.reshape(new_shape);
    };
    
    query = reshape_to_chunks(query);
    key = reshape_to_chunks(key);
    value = reshape_to_chunks(value);
    k_beta = reshape_to_chunks(k_beta);
    v_beta = reshape_to_chunks(v_beta);

    auto g_shape = g.sizes();
    std::vector<int64_t> g_new_shape = {
        g_shape[0], g_shape[1], 
        g_shape[2] / chunk_size, chunk_size
    };
    g = g.reshape(g_new_shape);
    auto mask = torch::triu(
        torch::ones({chunk_size, chunk_size}, torch::TensorOptions().dtype(torch::kBool).device(query.device())),
        0
    );

    g = g.cumsum(-1);
    auto g_diff = g.unsqueeze(-1) - g.unsqueeze(-2);
    auto decay_mask = g_diff.tril().exp().to(torch::kFloat32);
    decay_mask = decay_mask.tril();
    auto attn = -(torch::matmul(k_beta, key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0.0);
    for (int64_t i = 1; i < chunk_size; ++i) {
        if (!attn.is_contiguous()) {
          attn = attn.contiguous();
        }
        auto row = attn.slice(-2, i, i+1).slice(-1, 0, i).squeeze(-2).clone().contiguous();
        auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone().contiguous();
        auto row_unsq = row.unsqueeze(-1).contiguous();
        auto row_sub_mul = (row_unsq * sub).contiguous();
        auto row_sub_sum = row_sub_mul.sum(-2).contiguous();
        auto row_final = (row + row_sub_sum).contiguous();
        attn.index_put_(
            {
                torch::indexing::Ellipsis,
                torch::indexing::Slice(i, i+1),
                torch::indexing::Slice(0, i)
            },
            row_final.unsqueeze(-2)
        );
    }
    
    attn = attn + torch::eye(chunk_size, torch::TensorOptions().dtype(attn.dtype()).device(attn.device()));
    value = torch::matmul(attn, v_beta);
    auto k_cumdecay = torch::matmul(attn, (k_beta * g.exp().unsqueeze(-1)));
    torch::Tensor last_recurrent_state;
    if (!initial_state.has_value()) {
        last_recurrent_state = torch::zeros(
            {batch_size, num_heads, k_head_dim, v_head_dim},
            torch::TensorOptions().dtype(value.dtype()).device(value.device())
        );
    } else {
        last_recurrent_state = initial_state.value().to(value);
    }
    auto core_attn_out = torch::zeros_like(value);
    mask = torch::triu(
        torch::ones({chunk_size, chunk_size}, torch::TensorOptions().dtype(torch::kBool).device(query.device())),
        1
    );
    int64_t num_chunks = total_sequence_length / chunk_size;
    for (int64_t i = 0; i < num_chunks; ++i) {
        auto q_i = query.select(2, i);
        auto k_i = key.select(2, i);
        auto v_i = value.select(2, i);
        auto attn_i = (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_mask.select(2, i)).masked_fill_(mask, 0.0);
        auto v_prime = torch::matmul(k_cumdecay.select(2, i), last_recurrent_state);
        auto v_new = v_i - v_prime;
        auto attn_inter = torch::matmul(q_i * g.select(2, i).unsqueeze(-1).exp(), last_recurrent_state);
        core_attn_out.select(2, i) = attn_inter + torch::matmul(attn_i, v_new);
        auto g_i_last = g.select(2, i).select(-1, -1).unsqueeze(-1);
        auto g_exp_term = (g_i_last - g.select(2, i)).exp().unsqueeze(-1);
        auto k_g_exp = (k_i * g_exp_term).transpose(-1, -2).contiguous();
        last_recurrent_state =
            last_recurrent_state * g_i_last.unsqueeze(-1).exp() +
            torch::matmul(k_g_exp, v_new);
    }
    auto core_attn_out_shape = core_attn_out.sizes();
    std::vector<int64_t> reshape_shape = {
        core_attn_out_shape[0], core_attn_out_shape[1],
        core_attn_out_shape[2] * core_attn_out_shape[3],
        core_attn_out_shape[4]
    };
    core_attn_out = core_attn_out.reshape(reshape_shape);
    core_attn_out = core_attn_out.slice(2, 0, sequence_length);
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
    return std::make_tuple(core_attn_out, last_recurrent_state);
}
} // namespace

Qwen3NextGatedDeltaNetImpl::Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                                       const QuantArgs& quant_args,
                                       const ParallelArgs& parallel_args,
                                       const torch::TensorOptions& options) {
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());
  tp_size_ = parallel_args.tp_group_->world_size();  
  rank_ = parallel_args.tp_group_->rank();
  num_k_heads_ = args.linear_num_key_heads();
  num_v_heads_ = args.linear_num_value_heads();
  head_k_dim_ = args.linear_key_head_dim();
  head_v_dim_ = args.linear_value_head_dim(); 
  k_size_ = num_k_heads_ * head_k_dim_; 
  v_size_ = num_v_heads_ * head_v_dim_;
  conv_kernel_size_ = args.linear_conv_kernel_dim();
  // 0. QKVZ parallel linear
  conv1d_ = register_module("conv1d",
                            ColumnParallelLinear(args.linear_conv_kernel_dim(),
                                                k_size_ * 2 + v_size_,
                                                /*bias=*/false,
                                                /*gather_output=*/false,
                                                quant_args,
                                                parallel_args,
                                                options));


  // 1. QKVZ parallel linear
  qkvz_proj_ = register_module("in_proj_qkvz",
                                ColumnParallelLinear(args.hidden_size(),
                                                    k_size_ * 2 + v_size_ * 2,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args,
                                                    options));
  // 2. Output projection
  ba_proj_ = register_module("in_proj_ba",
                              ColumnParallelLinear(args.hidden_size(),
                                                  num_v_heads_ * 2,
                                                  /*bias=*/false,
                                                  /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args,
                                                    options));
  auto opts = options.dtype(torch::kFloat32);
  dt_bias_ = register_parameter("dt_bias", torch::ones({num_v_heads_ / tp_size_}, opts), /*requires_grad=*/false);

  A_log_ = register_parameter("A_log", torch::empty({num_v_heads_ / tp_size_}, opts), /*requires_grad=*/false);
  // 3. Output projection
  o_proj_ = register_module("out_proj",
                            RowParallelLinear(v_size_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args,
                                              options));

  // 4. RMSNorm
  norm_ = register_module("norm", RmsNormGated(head_v_dim_, args.rms_norm_eps(), options));

}

torch::Tensor Qwen3NextGatedDeltaNetImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {

  auto qkvz = qkvz_proj_->forward(hidden_states);
  auto qkvz_reshaped = reshape_qkvz_with_pad(attn_metadata, qkvz);
  auto [q, k, v, z] = process_qkvz_tensor(qkvz_reshaped);
  auto ba = ba_proj_->forward(hidden_states);
  auto ba_reshaped = reshape_qkvz_with_pad(attn_metadata, ba);
  auto [b, a] = process_ba_tensor(ba_reshaped);
  auto rearrange_merge = [](const torch::Tensor& t) {
    TORCH_CHECK(t.dim() > 2, "Tensor must have at least 2 dims! but got ", t.dim());
    std::vector<int64_t> new_shape;
    int64_t slice_end = t.dim() - 2;
    auto valid_slice = t.sizes().slice(0, slice_end);
    new_shape = std::vector<int64_t>(valid_slice.begin(), valid_slice.end());
    int64_t last_two_dim = t.size(slice_end) * t.size(slice_end + 1);
    new_shape.push_back(last_two_dim);
    return t.reshape(new_shape);
  };

  q = rearrange_merge(q);
  k = rearrange_merge(k);
  v = rearrange_merge(v);

  torch::Tensor mixed_qkv = torch::cat({q, k, v}, q.dim() - 1);
  mixed_qkv = mixed_qkv.transpose(1,2);
  int64_t seq_len = mixed_qkv.size(2); 
  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  torch::Tensor g, beta, core_attn_out, last_recurrent_state;
  auto device = mixed_qkv.device();
  auto conv_weight = conv1d_->weight();

  if (attn_metadata.is_prefill) {
    torch::Tensor conv_state = (seq_len < conv_kernel_size_-1) ? torch::pad(mixed_qkv, {0, conv_kernel_size_-1-seq_len}) : (seq_len > conv_kernel_size_-1) ? mixed_qkv.narrow(-1, seq_len-conv_kernel_size_+1, conv_kernel_size_-1): mixed_qkv;
    conv_cache.index_put_({input_params.block_tables.select(1,0)}, conv_state.to(conv_cache.dtype()));
    torch::Tensor bias;
    auto conv_output = torch::conv1d(
        mixed_qkv,
        conv_weight.unsqueeze(1).to(device),
        bias,  
        /*stride=*/std::vector<int64_t>{1},
        /*padding=*/std::vector<int64_t>{3},
        /*dilation=*/std::vector<int64_t>{1},
        /*groups=*/static_cast<int64_t>(mixed_qkv.size(1))
    );
    mixed_qkv = torch::silu(conv_output.slice(2,0,seq_len));

  } else {
    xllm::kernel::CausalConv1dUpdateParams params;
    params.x = mixed_qkv;
    params.conv_state = conv_cache;
    params.weight = conv_weight;
    params.conv_state_indices = attn_metadata.block_table.select(1,0);
    mixed_qkv = xllm::kernel::causal_conv1d_update(params);
  }

  if (attn_metadata.is_prefill) {
    beta = torch::sigmoid(b);
    torch::Tensor A_log_exp = A_log_.exp();
    torch::Tensor a_float = a.to(torch::kFloat32);
    torch::Tensor a_plus_dt = a_float + dt_bias_;
    torch::Tensor softplus_out = torch::nn::functional::softplus(
        a_plus_dt,
        torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0)
    );
    g = -A_log_exp * softplus_out;
    g = g.to(a.dtype()).contiguous();
  } else {
    xllm::kernel::FusedGdnGatingParams gdn_params;
    gdn_params.A_log = A_log_;
    gdn_params.a = a.view({-1, a.size(-1)});
    gdn_params.b = b.view({-1, b.size(-1)});
    gdn_params.dt_bias = dt_bias_;
    gdn_params.beta = 1.0f;
    gdn_params.threshold = 20.0f;
    std::tie(g, beta) = xllm::kernel::fused_gdn_gating(gdn_params);
  }

  auto [processed_q, processed_k, processed_v] = process_mixed_qkv(mixed_qkv);
  int64_t repeat_times = num_v_heads_ / num_k_heads_;
  if (repeat_times > 1) {
    processed_q = processed_q.repeat_interleave(repeat_times, 2);
    processed_k = processed_k.repeat_interleave(repeat_times, 2);
  }
  if (attn_metadata.is_prefill) {
    std::tie(core_attn_out, last_recurrent_state) = torch_chunk_gated_delta_rule(processed_q, processed_k, processed_v, g, beta);
    ssm_cache.index_put_({input_params.block_tables.select(1,0)}, last_recurrent_state.to(ssm_cache.dtype()));
  } else {
    auto ssm_state = torch::index_select(ssm_cache, 0, attn_metadata.block_table.select(1,0));
    std::tie(core_attn_out, last_recurrent_state) = torch_recurrent_gated_delta_rule(processed_q, processed_k, processed_v, g, beta, ssm_state);
    ssm_cache.index_put_({attn_metadata.block_table.select(1,0)}, last_recurrent_state.to(ssm_cache.dtype()));
  }

  auto z_reshaped = z.view({-1, z.size(-1)});
  auto core_attn_out_reshaped = core_attn_out.view({-1, core_attn_out.size(-1)});
  auto norm_out = norm_->forward(core_attn_out_reshaped, z_reshaped);
  auto z_shape_og = z.sizes().vec();
  norm_out = norm_out.view(z_shape_og);
  norm_out = norm_out.view({-1, norm_out.size(2), norm_out.size(3)});

  auto rearranged_norm = rearrange_merge(norm_out);
  rearranged_norm = reshape_qkvz_unpad(attn_metadata, rearranged_norm);
  auto attn_output = o_proj_->forward(rearranged_norm);
  return attn_output; 
}

torch::Tensor Qwen3NextGatedDeltaNetImpl::reshape_qkvz_unpad(const AttentionMetadata& attn_metadata, const torch::Tensor& padded_qkvz) {
    if (!attn_metadata.is_prefill) {
      return padded_qkvz;
    }
    std::vector<torch::Tensor> valid_batches;
    int64_t bs = attn_metadata.query_start_loc.size(0);
    int64_t max_len = attn_metadata.max_query_len;
    const auto& ori_seq_lens = attn_metadata.query_start_loc;
    auto reshaped_qkvz = padded_qkvz.view({bs, max_len, -1});
    for (int64_t b = 0; b < bs; ++b) {
        int64_t ori_len = ori_seq_lens[b].item<int64_t>();
        torch::Tensor valid_batch = reshaped_qkvz[b].slice(0, 0, ori_len);
        valid_batches.push_back(valid_batch);
    }
    return torch::cat(valid_batches, 0).contiguous();
}

torch::Tensor Qwen3NextGatedDeltaNetImpl::reshape_qkvz_with_pad(const AttentionMetadata& attn_metadata, const torch::Tensor& qkvz) {
    int64_t bs = attn_metadata.query_start_loc.size(0);
    int64_t max_len = attn_metadata.max_query_len;
    const auto& start_loc = attn_metadata.query_start_loc;
    if (!attn_metadata.is_prefill) {
      return qkvz.view({bs, -1, qkvz.size(-1)});
    }
    std::vector<torch::Tensor> batches;
    int64_t idx = 0;
    for (int64_t b = 0; b < bs; ++b) {
        int64_t cur_len = start_loc[b].item<int64_t>();
        torch::Tensor batch = qkvz.slice(0, idx, idx + cur_len).contiguous();
	idx = idx + cur_len;
        if (batch.size(0) != max_len) {
            batch = batch.size(0) > max_len 
                    ? batch.slice(0, 0, max_len).contiguous()
                    : torch::nn::functional::pad(
                        batch, 
                        torch::nn::functional::PadFuncOptions({0, 0, 0, max_len - batch.size(0)})
                      ).contiguous();
        }
        batches.push_back(batch);
    }
    auto ret = torch::stack(batches, 0).contiguous();
    return ret;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Qwen3NextGatedDeltaNetImpl::process_mixed_qkv(torch::Tensor& mixed_qkv) {
  mixed_qkv = mixed_qkv.transpose(1,2);
  int64_t batch_size = mixed_qkv.size(0);
  int64_t seq_len = mixed_qkv.size(1);
  std::vector<int64_t> split_sizes = {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};
  auto processed_qkv = torch::split(mixed_qkv, split_sizes, 2);
  auto processed_q = processed_qkv[0];
  auto processed_k = processed_qkv[1];
  auto processed_v = processed_qkv[2]; 
  processed_q = processed_q.view({batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_k = processed_k.view({batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_v = processed_v.view({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  return std::make_tuple(processed_q, processed_k, processed_v);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
Qwen3NextGatedDeltaNetImpl::process_qkvz_tensor(const torch::Tensor& qkvz) {
  
  std::vector<int64_t> new_tensor_shape_qkvz = [&]() {
    std::vector<int64_t> dims;
    dims.push_back(qkvz.size(0));
    if (qkvz.dim() >= 3) {
        dims.push_back(qkvz.size(1));
    }

    int64_t dim1 = num_k_heads_ / tp_size_;
    int64_t dim2 = head_k_dim_ + head_k_dim_ + (head_v_dim_ + head_v_dim_) * num_v_heads_ / num_k_heads_;
    dims.push_back(dim1);
    dims.push_back(dim2);

    return dims;
  }();

  auto reshaped_qkvz = qkvz.view(new_tensor_shape_qkvz);
  auto qkvz_split = torch::split(reshaped_qkvz, 
    {head_k_dim_, head_k_dim_, 
     num_v_heads_ * head_v_dim_ / num_k_heads_, 
     num_v_heads_ * head_v_dim_ / num_k_heads_}, reshaped_qkvz.dim()-1);
     
  auto q = qkvz_split[0].contiguous();
  auto k = qkvz_split[1].contiguous();
  auto v = qkvz_split[2].contiguous();
  auto z = qkvz_split[3].contiguous();
  

  v = v.view({v.size(0), v.size(1), -1, head_v_dim_});
  z = z.view({z.size(0), z.size(1), -1, head_v_dim_});

  return std::make_tuple(q, k, v, z);
}


std::tuple<torch::Tensor, torch::Tensor> 
Qwen3NextGatedDeltaNetImpl::process_ba_tensor(const torch::Tensor& ba) {

  std::vector<int64_t> new_tensor_shape_ba = [&]() {
    std::vector<int64_t> dims;
    dims.push_back(ba.size(0));
    dims.push_back(ba.size(1));
    int64_t dim1 = num_k_heads_ / tp_size_;
    int64_t dim2 = 2 * num_v_heads_ / num_k_heads_;
    dims.push_back(dim1);
    dims.push_back(dim2);
    return dims;
  }();

  auto reshaped_ba = ba.view(new_tensor_shape_ba);
  auto ba_split = torch::split(reshaped_ba, 
    {num_v_heads_ / num_k_heads_, num_v_heads_ / num_k_heads_}, reshaped_ba.dim()-1);
     
  auto b = ba_split[0].contiguous();
  auto a = ba_split[1].contiguous();

  b = b.reshape({b.size(0), b.size(1), num_v_heads_ / tp_size_});
  a = a.reshape({a.size(0), a.size(1), num_v_heads_ / tp_size_});
  
  return std::make_tuple(b, a);
}

void Qwen3NextGatedDeltaNetImpl::load_state_dict(const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = tp_size_;
  const int32_t shard_tensor_count = 3;
  const std::vector<int64_t> shard_sizes = {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};
  qkvz_proj_->load_state_dict(state_dict.get_dict_with_prefix("in_proj_qkvz."));
  ba_proj_->load_state_dict(state_dict.get_dict_with_prefix("in_proj_ba."));
  
  if (auto w = state_dict.get_tensor("conv1d.weight"); w.defined()) {
    conv1d_->load_state_dict(StateDict({{"weight", w.squeeze(1)}}), shard_tensor_count, shard_sizes);
  }
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
  if (auto w = state_dict.get_tensor("norm.weight"); w.defined()) {
    norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  LOAD_SHARDED_WEIGHT(dt_bias, 0);
  LOAD_SHARDED_WEIGHT(A_log, 0);
}

}  // namespace layer
}  // namespace xllm
