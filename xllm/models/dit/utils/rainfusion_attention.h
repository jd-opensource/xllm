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

#pragma once

#include <torch/torch.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "core/kernels/npu/npu_ops_api.h"

namespace xllm::dit {

struct RainFusionConfig {
  float sparsity = 0.5;
  int64_t pool_size = 128;
  int64_t inner_precise = 0;
  int64_t sparse_start_step = 0;
  int64_t mask_refresh_interval = 1;
  bool enabled = false;
};

// Per-request dynamic state for RainFusionV3 attention.
// Must NOT be stored as model layer members — each request owns its own
// instance, passed through the forward call chain.
struct RainFusionState {
  torch::Tensor cached_mask;
  int64_t current_step = 0;
  std::vector<int64_t> latent_shape = {1, 1, 1};
};

namespace {

torch::Tensor avgpool(const torch::Tensor& input,
                      int64_t pool_size,
                      const std::string& input_layout) {
  std::vector<torch::Tensor> pooled_parts;
  if (input_layout == "BSND") {
    int64_t batch = input.size(0);
    int64_t seqlen = input.size(1);
    int64_t headnum = input.size(2);
    int64_t dim = input.size(3);
    int64_t num_full_blocks = seqlen / pool_size;
    int64_t tail_size = seqlen % pool_size;

    if (num_full_blocks > 0) {
      auto full_blocks = input.slice(1, 0, num_full_blocks * pool_size);
      full_blocks = full_blocks.reshape(
          {batch, num_full_blocks, pool_size, headnum, dim});
      pooled_parts.emplace_back(full_blocks.mean(2));
    }
    if (tail_size > 0) {
      auto tail_block = input.slice(1, num_full_blocks * pool_size, seqlen);
      tail_block = tail_block.reshape({batch, 1, tail_size, headnum, dim});
      pooled_parts.emplace_back(tail_block.mean(2));
    }
  } else {
    // BNSD: [B, N, S, D]
    int64_t batch = input.size(0);
    int64_t headnum = input.size(1);
    int64_t seqlen = input.size(2);
    int64_t dim = input.size(3);
    int64_t num_full_blocks = seqlen / pool_size;
    int64_t tail_size = seqlen % pool_size;

    if (num_full_blocks > 0) {
      auto full_blocks = input.slice(2, 0, num_full_blocks * pool_size);
      full_blocks = full_blocks.reshape(
          {batch, headnum, num_full_blocks, pool_size, dim});
      pooled_parts.emplace_back(full_blocks.mean(3));
    }
    if (tail_size > 0) {
      auto tail_block = input.slice(2, num_full_blocks * pool_size, seqlen);
      tail_block = tail_block.reshape({batch, headnum, 1, tail_size, dim});
      pooled_parts.emplace_back(tail_block.mean(3));
    }
  }
  if (pooled_parts.size() > 1) {
    int64_t cat_dim = (input_layout == "BSND") ? 1 : 2;
    return torch::cat(pooled_parts, cat_dim);
  }
  return pooled_parts[0];
}

// Rearrange tokens from (f, h, w) spatial order to (f, hn, wn, 8, 8) block
// order. First frame is kept unchanged (first-frame protection).
torch::Tensor rearrange_with_remaining(const torch::Tensor& tensor,
                                       int64_t tq,
                                       int64_t hq,
                                       int64_t wq,
                                       const std::string& input_layout) {
  int64_t first_frame_len = hq * wq;
  int64_t frame_num = tq;

  if (hq % 8 == 0 && wq % 8 == 0) {
    // Aligned path
    int64_t hn = hq / 8;
    int64_t wn = wq / 8;
    if (input_layout == "BSND") {
      return tensor
          .reshape({tensor.size(0),
                    frame_num,
                    hn,
                    8,
                    wn,
                    8,
                    tensor.size(2),
                    tensor.size(3)})
          .permute({0, 1, 2, 4, 3, 5, 6, 7})
          .contiguous()
          .reshape({tensor.size(0), -1, tensor.size(2), tensor.size(3)});
    } else {
      // BNSD: [B, N, (f hn 8 wn 8), D] -> [B, N, (f hn wn 8 8), D]
      return tensor
          .reshape({tensor.size(0),
                    tensor.size(1),
                    frame_num,
                    hn,
                    8,
                    wn,
                    8,
                    tensor.size(3)})
          .permute({0, 1, 2, 3, 5, 4, 6, 7})
          .contiguous()
          .reshape({tensor.size(0), tensor.size(1), -1, tensor.size(3)});
    }
  }

  // Remainder path
  int64_t hq_block = (hq / 8) * 8;
  int64_t wq_block = (wq / 8) * 8;
  int64_t hq_rem = hq % 8;
  int64_t wq_rem = wq % 8;
  int64_t hn = hq_block / 8;
  int64_t wn = wq_block / 8;

  if (input_layout == "BSND") {
    auto tensor_first = tensor.slice(1, 0, first_frame_len);
    auto tensor_rest = tensor.slice(1, first_frame_len);
    auto tensor_hwt = tensor_rest.reshape({tensor.size(0),
                                           frame_num - 1,
                                           hq,
                                           wq,
                                           tensor.size(2),
                                           tensor.size(3)});
    torch::Tensor tensor_h_r, tensor_w_r;
    if (hq_rem > 0) {
      auto splits = tensor_hwt.split({hq_block, hq_rem}, 2);
      tensor_hwt = splits[0];
      tensor_h_r = splits[1].reshape(
          {tensor.size(0), frame_num - 1, -1, tensor.size(2), tensor.size(3)});
    }
    if (wq_rem > 0) {
      auto splits = tensor_hwt.split({wq_block, wq_rem}, 3);
      tensor_hwt = splits[0];
      tensor_w_r = splits[1].reshape(
          {tensor.size(0), frame_num - 1, -1, tensor.size(2), tensor.size(3)});
    }
    tensor_hwt = tensor_hwt
                     .reshape({tensor.size(0),
                               frame_num - 1,
                               hn,
                               8,
                               wn,
                               8,
                               tensor.size(2),
                               tensor.size(3)})
                     .permute({0, 1, 2, 4, 3, 5, 6, 7})
                     .contiguous()
                     .reshape({tensor.size(0),
                               frame_num - 1,
                               hn * wn * 64,
                               tensor.size(2),
                               tensor.size(3)});
    if (hq_rem > 0) {
      tensor_hwt = torch::cat({tensor_hwt, tensor_h_r}, 2);
    }
    if (wq_rem > 0) {
      tensor_hwt = torch::cat({tensor_hwt, tensor_w_r}, 2);
    }
    tensor_hwt = tensor_hwt.reshape(
        {tensor.size(0), -1, tensor.size(2), tensor.size(3)});
    return torch::cat({tensor_first, tensor_hwt}, 1);
  } else {
    // BNSD
    auto tensor_first = tensor.slice(2, 0, first_frame_len);
    auto tensor_rest = tensor.slice(2, first_frame_len);
    auto tensor_hwt = tensor_rest.reshape({tensor.size(0),
                                           tensor.size(1),
                                           frame_num - 1,
                                           hq,
                                           wq,
                                           tensor.size(3)});
    torch::Tensor tensor_h_r, tensor_w_r;
    if (hq_rem > 0) {
      auto splits = tensor_hwt.split({hq_block, hq_rem}, 3);
      tensor_hwt = splits[0];
      tensor_h_r = splits[1].reshape(
          {tensor.size(0), tensor.size(1), frame_num - 1, -1, tensor.size(3)});
    }
    if (wq_rem > 0) {
      auto splits = tensor_hwt.split({wq_block, wq_rem}, 4);
      tensor_hwt = splits[0];
      tensor_w_r = splits[1].reshape(
          {tensor.size(0), tensor.size(1), frame_num - 1, -1, tensor.size(3)});
    }
    tensor_hwt = tensor_hwt
                     .reshape({tensor.size(0),
                               tensor.size(1),
                               frame_num - 1,
                               hn,
                               8,
                               wn,
                               8,
                               tensor.size(3)})
                     .permute({0, 1, 2, 3, 5, 4, 6, 7})
                     .contiguous()
                     .reshape({tensor.size(0),
                               tensor.size(1),
                               frame_num - 1,
                               hn * wn * 64,
                               tensor.size(3)});
    if (hq_rem > 0) {
      tensor_hwt = torch::cat({tensor_hwt, tensor_h_r}, 3);
    }
    if (wq_rem > 0) {
      tensor_hwt = torch::cat({tensor_hwt, tensor_w_r}, 3);
    }
    tensor_hwt = tensor_hwt.reshape(
        {tensor.size(0), tensor.size(1), -1, tensor.size(3)});
    return torch::cat({tensor_first, tensor_hwt}, 2);
  }
}

// Compute block sparse mask from pooled Q/K/V (concatenated along dim=0).
// Returns int8 binary mask of shape [B, N, q_blocks, kv_blocks].
torch::Tensor get_blockwise_mask(const torch::Tensor& q_pool,
                                 const torch::Tensor& k_pool,
                                 int64_t txt_len,
                                 float sparsity,
                                 double scale,
                                 int64_t pool_size,
                                 int64_t tq,
                                 int64_t hq,
                                 int64_t wq,
                                 const std::string& input_layout,
                                 bool protect_first_frame) {
  // Each head computes its own mask independently
  auto q_for_mask = q_pool;
  auto k_for_mask = k_pool;

  // Attention scores: Q @ K^T * scale
  torch::Tensor attn_scores;
  if (input_layout == "BSND") {
    attn_scores =
        torch::einsum("blnd,bsnd->bnls", {q_for_mask, k_for_mask}) * scale;
  } else {
    attn_scores =
        torch::einsum("bnld,bnsd->bnls", {q_for_mask, k_for_mask}) * scale;
  }

  auto score_matrix = torch::softmax(attn_scores, -1);
  int64_t cols = score_matrix.size(-1);
  int64_t keep_len = std::max(
      static_cast<int64_t>(std::ceil(cols * (1.0 - sparsity))), int64_t(1));

  auto topk_result = score_matrix.topk(keep_len, -1);
  auto thresholds = std::get<0>(topk_result).slice(-1, keep_len - 1, keep_len);
  auto mask = score_matrix >= thresholds;

  // Text token protection
  int64_t text_block_num = (txt_len + pool_size - 1) / pool_size;
  if (text_block_num > 0) {
    mask.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(),
         torch::indexing::Slice(-text_block_num, torch::indexing::None),
         torch::indexing::Slice()},
        true);
    mask.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(),
         torch::indexing::Slice(),
         torch::indexing::Slice(-text_block_num, torch::indexing::None)},
        true);
  }

  // First frame protection
  if (protect_first_frame) {
    int64_t first_frame_len = hq * wq;
    int64_t firstframe_block_num =
        (first_frame_len + pool_size - 1) / pool_size;
    if (firstframe_block_num > 0) {
      mask.index_put_({torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(0, firstframe_block_num),
                       torch::indexing::Slice()},
                      true);
      mask.index_put_({torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(0, firstframe_block_num)},
                      true);
    }
  }

  auto mask_int8 = mask.to(torch::kInt8);
  return mask_int8;
}

// Inverse of rearrange_with_remaining. Converts from (f, hn, wn, 8, 8) block
// order back to (f, h, w) spatial order.
torch::Tensor bsa_inv_rearrange(const torch::Tensor& out,
                                int64_t tq,
                                int64_t hq,
                                int64_t wq,
                                const std::string& input_layout) {
  int64_t hn = hq / 8;
  int64_t wn = wq / 8;

  if (hq % 8 == 0 && wq % 8 == 0) {
    if (input_layout == "BNSD") {
      int64_t n = out.size(1);
      return out.reshape({out.size(0), n, tq, hn, wn, 8, 8, out.size(3)})
          .permute({0, 1, 2, 3, 5, 4, 6, 7})
          .contiguous()
          .reshape({out.size(0), n, tq * hq * wq, out.size(3)});
    } else {
      int64_t n = out.size(2);
      return out.reshape({out.size(0), tq, hn, wn, 8, 8, n, out.size(3)})
          .permute({0, 1, 2, 4, 3, 5, 6, 7})
          .contiguous()
          .reshape({out.size(0), tq * hq * wq, n, out.size(3)});
    }
  }

  // Remainder path
  int64_t first_frame_len = hq * wq;
  int64_t hq_block = (hq / 8) * 8;
  int64_t wq_block = (wq / 8) * 8;
  int64_t hq_rem = hq % 8;
  int64_t wq_rem = wq % 8;
  int64_t block_size = hn * wn * 64;
  int64_t h_rem_size = hq_rem * wq;

  if (input_layout == "BNSD") {
    int64_t n = out.size(1);
    auto out_first = out.slice(2, 0, first_frame_len);
    auto out_rest = out.slice(2, first_frame_len);
    out_rest = out_rest.reshape({out.size(0), n, tq - 1, hq * wq, out.size(3)});

    auto t_block = out_rest.slice(3, 0, block_size);
    torch::Tensor t_h_r, t_w_r;
    if (hq_rem > 0) {
      t_h_r = out_rest.slice(3, block_size, block_size + h_rem_size);
    }
    if (wq_rem > 0) {
      t_w_r = out_rest.slice(3, block_size + h_rem_size);
    }

    t_block =
        t_block.reshape({out.size(0), n, tq - 1, hn, wn, 8, 8, out.size(3)})
            .permute({0, 1, 2, 3, 5, 4, 6, 7})
            .contiguous()
            .reshape({out.size(0), n, tq - 1, hq_block, wq_block, out.size(3)});
    if (wq_rem > 0) {
      t_block = torch::cat(
          {t_block,
           t_w_r.reshape(
               {out.size(0), n, tq - 1, hq_block, wq_rem, out.size(3)})},
          4);
    }
    if (hq_rem > 0) {
      t_block = torch::cat(
          {t_block,
           t_h_r.reshape({out.size(0), n, tq - 1, hq_rem, wq, out.size(3)})},
          3);
    }
    auto out_rest_merged =
        t_block.reshape({out.size(0), n, (tq - 1) * hq * wq, out.size(3)});
    return torch::cat({out_first, out_rest_merged}, 2);
  } else {
    // BSND
    int64_t n = out.size(2);
    auto out_first = out.slice(1, 0, first_frame_len);
    auto out_rest = out.slice(1, first_frame_len);
    out_rest = out_rest.reshape({out.size(0), tq - 1, hq * wq, n, out.size(3)});

    auto t_block = out_rest.slice(2, 0, block_size);
    torch::Tensor t_h_r, t_w_r;
    if (hq_rem > 0) {
      t_h_r = out_rest.slice(2, block_size, block_size + h_rem_size);
    }
    if (wq_rem > 0) {
      t_w_r = out_rest.slice(2, block_size + h_rem_size);
    }

    t_block =
        t_block.reshape({out.size(0), tq - 1, hn, wn, 8, 8, n, out.size(3)})
            .permute({0, 1, 2, 4, 3, 5, 6, 7})
            .contiguous()
            .reshape({out.size(0), tq - 1, hq_block, wq_block, n, out.size(3)});
    if (wq_rem > 0) {
      t_block = torch::cat(
          {t_block,
           t_w_r.reshape(
               {out.size(0), tq - 1, hq_block, wq_rem, n, out.size(3)})},
          3);
    }
    if (hq_rem > 0) {
      t_block = torch::cat(
          {t_block,
           t_h_r.reshape({out.size(0), tq - 1, hq_rem, wq, n, out.size(3)})},
          2);
    }
    auto out_rest_merged =
        t_block.reshape({out.size(0), (tq - 1) * hq * wq, n, out.size(3)});
    return torch::cat({out_first, out_rest_merged}, 1);
  }
}

}  // namespace

// RainFusionV3 block sparse attention — main entry point.
// query/key/value are in BNSD layout [B, N, S, D].
// state holds per-request dynamic state (cached_mask, current_step,
// latent_shape).
std::tuple<torch::Tensor, torch::Tensor> rainfusion_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const RainFusionConfig& config,
    RainFusionState& state) {
  CHECK(query.dim() == 4) << "query must be 4D [B, N, S, D]";
  CHECK(query.dtype() == torch::kHalf || query.dtype() == torch::kBFloat16)
      << "query must be fp16 or bf16";
  CHECK(config.sparsity >= 0.0f && config.sparsity < 1.0f)
      << "sparsity must be in [0.0, 1.0)";
  CHECK(config.pool_size > 0 && config.pool_size % 128 == 0)
      << "pool_size must be positive multiple of 128";
  int64_t tq = state.latent_shape[0];
  int64_t hq = state.latent_shape[1];
  int64_t wq = state.latent_shape[2];
  int64_t num_heads = query.size(1);
  double scale = std::pow(static_cast<double>(query.size(-1)), -0.5);

  // Step 1: Rearrange Q, K, V individually to avoid 3× copy from torch::cat.
  auto q_rearranged = rearrange_with_remaining(query, tq, hq, wq, "BNSD");
  auto k_rearranged = rearrange_with_remaining(key, tq, hq, wq, "BNSD");
  auto v_rearranged = rearrange_with_remaining(value, tq, hq, wq, "BNSD");

  // Step 2-3: Pooling + mask generation (skip if using cached mask)
  torch::Tensor new_mask;
  bool use_cached = state.cached_mask.defined() &&
                    (config.mask_refresh_interval <= 0 ||
                     (state.current_step % config.mask_refresh_interval != 0));
  if (!use_cached) {
    auto qk = torch::cat({q_rearranged, k_rearranged}, 0);
    auto qk_pool = avgpool(qk, config.pool_size, "BNSD");
    auto chunks = qk_pool.chunk(2, 0);
    // NOTE: txt_len=0 because RainFusion currently only runs on self-attention
    // (pre-cross-attention path). Text token protection is reserved for future
    // text-conditioned video generation where txt_len > 0.
    new_mask = get_blockwise_mask(chunks[0],
                                  chunks[1],
                                  /*txt_len=*/0,
                                  config.sparsity,
                                  scale,
                                  config.pool_size,
                                  tq,
                                  hq,
                                  wq,
                                  "BNSD",
                                  /*protect_first_frame=*/true);
  } else {
    new_mask = state.cached_mask;
  }

  // Step 4: Block sparse attention via aclnn
  int64_t batch_size = query.size(0);

  // Per-batch actual sequence lengths after spatial rearrangement.
  // The rearranged tensor packs (f * hn * wn * 64) tokens per batch;
  // passing actual lengths lets the BSA kernel handle tail blocks correctly.
  std::vector<int64_t> actual_seq_lens(batch_size,
                                       q_rearranged.size(/*S dim in BNSD=*/2));

  auto [attn_out, lse] = xllm::kernel::npu::npu_block_sparse_attention(
      q_rearranged,
      k_rearranged,
      v_rearranged,
      new_mask,
      {config.pool_size, config.pool_size},
      "BNSD",
      "BNSD",
      num_heads,
      scale,
      config.inner_precise,
      /*softmax_lse_flag=*/0,
      std::optional<torch::IntArrayRef>(actual_seq_lens),
      std::optional<torch::IntArrayRef>(actual_seq_lens));

  // Step 5: Inverse spatial rearrangement
  auto out = bsa_inv_rearrange(attn_out, tq, hq, wq, "BNSD");

  return std::make_tuple(out, new_mask);
}

}  // namespace xllm::dit
