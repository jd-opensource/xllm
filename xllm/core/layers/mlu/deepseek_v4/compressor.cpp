/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "layers/mlu/deepseek_v4/compressor.h"

#include <glog/logging.h>

#include <cmath>
#include <limits>

#include "torch/torch.h"

namespace {

bool is_power_of_two(int64_t n) { return n > 0 && ((n & (n - 1)) == 0); }

torch::Tensor create_hadamard_matrix(int64_t n,
                                     torch::Dtype dtype,
                                     torch::Device device,
                                     bool normalize) {
  CHECK(is_power_of_two(n)) << "hadamard_matrix: n must be a power of two.";

  const torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).device(device);
  torch::Tensor h = torch::ones({1, 1}, options);
  for (int64_t m = 1; m < n; m <<= 1) {
    torch::Tensor top = torch::cat({h, h}, /*dim=*/1);
    torch::Tensor bottom = torch::cat({h, -h}, /*dim=*/1);
    h = torch::cat({top, bottom}, /*dim=*/0);
  }

  if (normalize) {
    h = h / std::sqrt(static_cast<double>(n));
  }
  return h;
}

torch::Tensor hadamard_transform_ref(const torch::Tensor& x,
                                     const torch::Tensor& h_matrix,
                                     double scale) {
  const auto x_shape = x.sizes();
  const int64_t dim = x.size(-1);
  torch::Tensor x_2d = x.reshape({-1, dim});

  const double log_dim = std::ceil(std::log2(static_cast<double>(dim)));
  const int64_t dim_padded =
      static_cast<int64_t>(1ULL << static_cast<uint64_t>(log_dim));
  if (dim != dim_padded) {
    x_2d = torch::nn::functional::pad(
        x_2d,
        torch::nn::functional::PadFuncOptions({0, dim_padded - dim})
            .mode(torch::kConstant)
            .value(0));
  }

  torch::Tensor x_float = x_2d.to(torch::kFloat32);
  torch::Tensor h_float = h_matrix.to(torch::kFloat32);
  torch::Tensor out_full = torch::matmul(x_float, h_float.t());

  if (std::abs(scale - 1.0) > 1e-9) {
    out_full.mul_(scale);
  }

  torch::Tensor out = out_full.index(
      {torch::indexing::Slice(), torch::indexing::Slice(0, dim)});
  return out.reshape(x_shape).to(x.dtype());
}

torch::Tensor rotate_activation(const torch::Tensor& input,
                                const torch::Tensor& hadamard_matrix) {
  CHECK(input.dtype() == torch::kBFloat16)
      << "rotate_activation: input must be bfloat16.";
  const int64_t hidden_size = input.size(-1);
  const double scale = std::pow(static_cast<double>(hidden_size), -0.5);
  return hadamard_transform_ref(input, hadamard_matrix, scale);
}

}  // namespace

namespace xllm {
namespace layer {

// Apply rotary position embedding by treating last dim as complex numbers
void apply_rotary_emb(torch::Tensor& x,
                      const torch::Tensor& freqs_cis,
                      bool inverse) {
  CHECK(x.size(-1) % 2 == 0) << "apply_rotary_emb: last dimension must be even";

  // Convert to complex representation: [*, D] -> [*, D/2]
  auto x_float = x.to(torch::kFloat32);
  auto x_complex =
      torch::view_as_complex(x_float.unflatten(-1, {-1, 2}).contiguous());
  auto freqs = inverse ? freqs_cis.conj() : freqs_cis;

  // Broadcast freqs to match x dimensions
  torch::Tensor freqs_reshaped;
  if (x_complex.dim() == 2) {
    freqs_reshaped = freqs.view({x_complex.size(0), x_complex.size(-1)});
  } else {
    freqs_reshaped = freqs.view({x_complex.size(0), 1, x_complex.size(-1)});
  }

  // Complex multiplication and convert back to real
  auto x_out = torch::view_as_real(x_complex * freqs_reshaped).flatten(-2);
  x.copy_(x_out);
}

CompressorImpl::CompressorImpl(int64_t dim,
                               int64_t head_dim,
                               int64_t rope_head_dim,
                               int64_t compress_ratio,
                               int64_t cached_state_num,
                               double norm_eps,
                               bool rotate,
                               DeepseekScalingRotaryEmbedding& rotary_emb,
                               const torch::TensorOptions& options)
    : dim_(dim),
      head_dim_(head_dim),
      rope_head_dim_(rope_head_dim),
      compress_ratio_(compress_ratio),
      overlap_(compress_ratio == 4),
      coeff_(1 + static_cast<int64_t>(overlap_)),
      rotate_(rotate),
      converted_(false),
      norm_eps_(norm_eps),
      rotary_emb_(rotary_emb) {
  ape_ = register_buffer("ape",
                         torch::empty({compress_ratio, coeff_ * head_dim},
                                      options.dtype(torch::kFloat32)));

  const int64_t out_dim = coeff_ * head_dim_;
  QuantArgs quant_args;

  wkv_ = register_module("wkv",
                         ReplicatedLinear(dim_,
                                          out_dim,
                                          /*bias=*/false,
                                          quant_args,
                                          options.dtype(torch::kFloat32)));
  wgate_ = register_module("wgate",
                           ReplicatedLinear(dim_,
                                            out_dim,
                                            /*bias=*/false,
                                            quant_args,
                                            options.dtype(torch::kFloat32)));
  norm_ = register_module("norm", RMSNorm(head_dim_, norm_eps_, options));

  const int64_t state_ratio = coeff_ * compress_ratio_;
  const int64_t state_dim = coeff_ * head_dim_;
  kv_state_ =
      register_buffer("kv_state",
                      torch::zeros({cached_state_num, state_ratio, state_dim},
                                   options.dtype(torch::kFloat32)));
  score_state_ =
      register_buffer("score_state",
                      torch::full({cached_state_num, state_ratio, state_dim},
                                  -std::numeric_limits<float>::infinity(),
                                  options.dtype(torch::kFloat32)));

  if (rotate_) {
    int64_t head_dim_padded = static_cast<int64_t>(
        std::pow(2, std::ceil(std::log2(static_cast<double>(head_dim_)))));
    hadamard_matrix_ =
        create_hadamard_matrix(
            head_dim_padded, torch::kFloat32, torch::kCPU, false)
            .to(options.device(), options.dtype().toScalarType());
  }
}

// Swap halves of APE for overlap mode (done once)
void CompressorImpl::convert_ape_if_needed() {
  if (overlap_ && !converted_) {
    auto ape_chunks = ape_.chunk(/*chunks=*/2, /*dim=*/-1);
    ape_.copy_(torch::cat({ape_chunks[1], ape_chunks[0]}, /*dim=*/-1));
    converted_ = true;
  }
}

void CompressorImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(ape);
  wkv_->load_state_dict(state_dict.get_dict_with_prefix("wkv."));
  wgate_->load_state_dict(state_dict.get_dict_with_prefix("wgate."));
  norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
}

// Apply RoPE to the last rope_head_dim dimensions of compressed KV
void CompressorImpl::apply_rope_to_compressed_kv(
    torch::Tensor& kv,
    const torch::Tensor& cutoff_positions) {
  // Extract rotary embedding part, with extra unsqueeze(-2)
  torch::Tensor kv_rope =
      kv.slice(-1, head_dim_ - rope_head_dim_, head_dim_).unsqueeze(-2);
  torch::Tensor cu_query_lens = torch::tensor(
      {0, static_cast<int32_t>(cutoff_positions.size(0))},
      torch::dtype(torch::kInt32).device(cutoff_positions.device()));
  rotary_emb_->forward(kv_rope,
                       cutoff_positions,
                       cu_query_lens,
                       /*is_prompt=*/false,
                       /*enforce_discrete=*/true);
}

void CompressorImpl::update_state_indices(int64_t state_idx,
                                          int64_t offset,
                                          const torch::Tensor& kv_remainder,
                                          const torch::Tensor& score_remainder,
                                          int64_t remainder) {
  if (remainder > 0) {
    kv_state_[state_idx].slice(0, offset, offset + remainder) = kv_remainder;
    score_state_[state_idx].slice(0, offset, offset + remainder) =
        score_remainder;
  }
}

// Transform tensor for overlap compression: [B, S, R, 2D] -> [B, S, 2R, D]
// Rearranges data so overlapping windows can be merged with previous tokens
torch::Tensor CompressorImpl::overlap_transform(const torch::Tensor& tensor,
                                                float fill_value) {
  const int64_t s = tensor.size(1);
  const int64_t ratio = compress_ratio_;
  const int64_t d = head_dim_;
  using torch::indexing::Slice;

  auto new_tensor = torch::full(
      {tensor.size(0), s, 2 * ratio, d}, fill_value, tensor.options());
  // Place second half into upper ratio slots
  new_tensor.index_put_(
      {Slice(), Slice(), Slice(ratio, 2 * ratio), Slice()},
      tensor.index({Slice(), Slice(), Slice(), Slice(d, 2 * d)}));

  // Shift first half from previous position into lower ratio slots
  if (s > 1) {
    auto shifted =
        tensor.index({Slice(), Slice(0, s - 1), Slice(), Slice(0, d)});
    new_tensor.index_put_({Slice(), Slice(1, s), Slice(0, ratio), Slice()},
                          shifted);
  }

  return new_tensor;
}

// Merge overlapping windows: combine previous and current ratio tokens
torch::Tensor CompressorImpl::compress_with_overlap(int64_t state_idx) {
  const int64_t ratio = compress_ratio_;
  const int64_t head = coeff_ * head_dim_;

  // Concatenate: [0:ratio]'s first half + [ratio:2*ratio]'s second half
  auto kv_overlap = torch::cat(
      {kv_state_[state_idx].slice(0, 0, ratio).slice(1, 0, head_dim_),
       kv_state_[state_idx]
           .slice(0, ratio, 2 * ratio)
           .slice(1, head_dim_, head)},
      0);
  auto score_overlap = torch::cat(
      {score_state_[state_idx].slice(0, 0, ratio).slice(1, 0, head_dim_),
       score_state_[state_idx]
           .slice(0, ratio, 2 * ratio)
           .slice(1, head_dim_, head)},
      0);

  // Weighted sum with softmax scores
  auto compressed_kv =
      (kv_overlap * torch::softmax(score_overlap, 0)).sum(0, /*keepdim=*/true);

  // Shift state: move [ratio:2*ratio] to [0:ratio]
  kv_state_[state_idx].slice(0, 0, ratio) =
      kv_state_[state_idx].slice(0, ratio, 2 * ratio);
  score_state_[state_idx].slice(0, 0, ratio) =
      score_state_[state_idx].slice(0, ratio, 2 * ratio);

  return compressed_kv;
}

// Compress input tokens into compact KV representations
// Returns: (compressed_kvs, compressed_lens) for each sequence in batch
std::tuple<std::vector<torch::Tensor>, std::vector<int64_t>>
CompressorImpl::forward(const torch::Tensor& x,
                        const torch::Tensor& positions,
                        const torch::Tensor& block_tables,
                        const torch::Tensor& query_start_loc,
                        const torch::Tensor& seq_lens,
                        const std::vector<int64_t>& batch_to_kv_state,
                        torch::Tensor& kv_cache,
                        int64_t window_offset,
                        const torch::Tensor& freqs_cis) {
  convert_ape_if_needed();

  const auto dtype = x.dtype();
  const auto x_float = x.to(torch::kFloat32);

  // Project input to KV and gate scores
  torch::Tensor kv_pack = wkv_->forward(x_float);
  torch::Tensor score_pack = wgate_->forward(x_float);

  std::vector<torch::Tensor> compress_kvs;
  std::vector<int64_t> compress_lens;

  // Calculate start position for each sequence
  auto query_lens =
      query_start_loc.slice(0, 1) - query_start_loc.slice(0, 0, -1);
  auto start_positions = seq_lens - query_lens;

  CHECK(batch_to_kv_state.size() == start_positions.size(0))
      << "batch_to_kv_state should have the same length as start_positions";

  const int64_t batch_size = start_positions.size(0);
  const auto options = torch::TensorOptions().device(x.device()).dtype(dtype);

  // Process each sequence independently
  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t query_idx_start = query_start_loc[i].item<int64_t>();
    const int64_t query_idx_end = query_start_loc[i + 1].item<int64_t>();
    const int64_t start_pos = start_positions[i].item<int64_t>();
    const int64_t state_idx = batch_to_kv_state[i];
    const int64_t seqlen = query_idx_end - query_idx_start;

    // skip sequences that are not in the state manager
    if (state_idx < 0) {
      compress_kvs.push_back(torch::empty({0, 1, head_dim_}, options));
      compress_lens.push_back(0);
      continue;
    }

    // Extract KV and scores for this sequence
    torch::Tensor kv = kv_pack.slice(0, query_idx_start, query_idx_end);
    torch::Tensor score = score_pack.slice(0, query_idx_start, query_idx_end);

    bool should_compress = false;
    torch::Tensor compressed_kv;

    // === PREFILL PHASE: compress entire sequence ===
    if (start_pos == 0) {
      should_compress = seqlen >= compress_ratio_;
      const int64_t remainder = seqlen % compress_ratio_;
      const int64_t cutoff = seqlen - remainder;
      const int64_t offset = overlap_ ? compress_ratio_ : 0;

      // Overlap mode: cache last compress_ratio tokens for overlapping with
      // next batch
      if (overlap_ && cutoff >= compress_ratio_) {
        auto kv_tail = kv.slice(0, cutoff - compress_ratio_, cutoff);
        auto score_tail = score.slice(0, cutoff - compress_ratio_, cutoff);
        kv_state_[state_idx].slice(0, 0, compress_ratio_) = kv_tail;
        score_state_[state_idx].slice(0, 0, compress_ratio_) =
            score_tail + ape_;
      }

      // Store remainder tokens that cannot form a complete compression window
      if (remainder > 0) {
        auto kv_parts = kv.split({cutoff, remainder}, 0);
        auto score_parts = score.split({cutoff, remainder}, 0);
        update_state_indices(state_idx,
                             offset,
                             kv_parts[1],
                             score_parts[1] + ape_.slice(0, 0, remainder),
                             remainder);
        kv = kv_parts[0];
        score = score_parts[0];
      }

      // Compress: [N*ratio, D] -> [N, ratio, D] -> weighted sum -> [N, D]
      auto kv_reshaped = kv.view({-1, compress_ratio_, coeff_ * head_dim_});
      auto score_reshaped =
          score.view({-1, compress_ratio_, coeff_ * head_dim_}) + ape_;

      // Transform for overlapping windows if needed
      if (overlap_) {
        kv_reshaped = overlap_transform(kv_reshaped.unsqueeze(0), /*filled*/ 0)
                          .squeeze(0);
        score_reshaped = overlap_transform(
                             score_reshaped.unsqueeze(0),
                             /*filled*/ -std::numeric_limits<float>::infinity())
                             .squeeze(0);
      }

      // Weighted sum with softmax attention
      compressed_kv = (kv_reshaped * torch::softmax(score_reshaped, 1)).sum(1);

      // === DECODE PHASE: accumulate tokens, compress when window is full ===
    } else {
      // Compress when we've accumulated compress_ratio tokens
      should_compress = ((start_pos + 1) % compress_ratio_ == 0);
      const int64_t pos_mod = start_pos % compress_ratio_;
      score = score + ape_[pos_mod];

      auto kv_squeezed = kv.squeeze(0);
      auto score_squeezed = score.squeeze(0);

      if (overlap_) {
        // Store in second half [ratio:2*ratio] for overlapping
        kv_state_[state_idx][compress_ratio_ + pos_mod] = kv_squeezed;
        score_state_[state_idx][compress_ratio_ + pos_mod] = score_squeezed;

        if (should_compress) {
          compressed_kv = compress_with_overlap(state_idx);
        }
      } else {
        // Simple mode: store in circular buffer [0:ratio]
        kv_state_[state_idx][pos_mod] = kv_squeezed;
        score_state_[state_idx][pos_mod] = score_squeezed;

        if (should_compress) {
          auto score_softmax = torch::softmax(score_state_[state_idx], 0);
          compressed_kv = (kv_state_[state_idx] * score_softmax).sum(0, true);
        }
      }
    }

    // Skip if no compression happened
    if (!should_compress) {
      compress_kvs.push_back(torch::empty({0, 1, head_dim_}, options));
      compress_lens.push_back(0);
      continue;
    }

    // Post-process: normalize and apply position encoding
    compressed_kv = compressed_kv.to(dtype);
    auto [normalized, _] = norm_->forward(compressed_kv);
    compressed_kv = normalized;

    // Extract positions for current sequence
    auto seq_positions = positions.slice(0, query_idx_start, query_idx_end);
    const int64_t cutoff = seqlen - (seqlen % compress_ratio_);
    torch::Tensor cutoff_positions;
    if (start_pos == 0) {
      // Prefill: pos[:cutoff:ratio]
      cutoff_positions =
          seq_positions.slice(0, 0, cutoff, compress_ratio_).contiguous();
    } else {
      // Decode: pos + 1 - compress_ratio
      cutoff_positions = seq_positions + 1 - compress_ratio_;
    }

    apply_rope_to_compressed_kv(compressed_kv, cutoff_positions);

    // Optional Hadamard rotation for better mixing
    if (rotate_) {
      compressed_kv = rotate_activation(compressed_kv, hadamard_matrix_);
    }

    // Write compressed KV to paged cache
    // here we must make sure the dtype of compressed_kv is the same as the
    // dtype of kv_cache
    if (start_pos == 0) {
      // Prefill: write all compressed tokens
      auto indices = block_tables.index(
          {i,
           torch::indexing::Slice(window_offset,
                                  window_offset + compressed_kv.size(0))});
      kv_cache.index_put_(
          {indices},
          compressed_kv.unsqueeze(1).unsqueeze(1).to(kv_cache.dtype()));
    } else {
      // Decode: write only one position
      auto index =
          block_tables.index({i, window_offset + start_pos / compress_ratio_});
      kv_cache.index_put_(
          {index}, compressed_kv.reshape({1, 1, 1, -1}).to(kv_cache.dtype()));
    }

    compress_kvs.push_back(compressed_kv.unsqueeze(-2));
    compress_lens.push_back(compressed_kv.size(0));
  }

  return {compress_kvs, compress_lens};
}

}  // namespace layer
}  // namespace xllm
