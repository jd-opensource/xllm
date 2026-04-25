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

#include "dsa_metadata_builder.h"

#include <algorithm>
#include <cmath>

#include "attention_metadata.h"
#include "attention_metadata_builder.h"
#include "dsa_metadata.h"
#include "framework/model/model_input_params.h"
#include "util/tensor_helper.h"

namespace xllm::layer {

AttentionMetadata DSAMetadataBuilder::build(
    const ModelInputParams& params,
    const torch::Tensor& positions,
    const torch::Tensor& dsa_cos_sin,
    const std::vector<std::vector<DSACacheInfo>>& caches_info,
    const std::vector<DSAGroupInfo>& group_infos,
    const torch::Tensor& dsa_c4_cos_sin,
    const torch::Tensor& dsa_c128_cos_sin) {
  // 1. Build base AttentionMetadata (q_cu_seq_lens, block_table, etc.)
  AttentionMetadata attn_metadata = AttentionMetadataBuilder::build(params);

  // 2. Build DSA-specific fields
  auto dsa_metadata = std::make_shared<DSAMetadata>();
  build_dsa_fields(params,
                   positions,
                   dsa_cos_sin,
                   dsa_c4_cos_sin,
                   dsa_c128_cos_sin,
                   caches_info,
                   group_infos,
                   *dsa_metadata);

  // 3. Keep DSA metadata independent while syncing base attention tensors.
  if (attn_metadata.attn_mask.defined()) {
    dsa_metadata->attn_mask = attn_metadata.attn_mask.clone();
  }

  if (attn_metadata.mrope_cos.defined() && !dsa_metadata->cos_table.defined()) {
    dsa_metadata->cos_table = attn_metadata.mrope_cos;
  }
  if (attn_metadata.mrope_sin.defined() && !dsa_metadata->sin_table.defined()) {
    dsa_metadata->sin_table = attn_metadata.mrope_sin;
  }

  // 4. Attach to AttentionMetadata
  attn_metadata.dsa_metadata = std::move(dsa_metadata);

  return attn_metadata;
}

void DSAMetadataBuilder::build_dsa_fields(
    const ModelInputParams& params,
    const torch::Tensor& positions,
    const torch::Tensor& dsa_cos_sin,
    const torch::Tensor& dsa_c4_cos_sin,
    const torch::Tensor& dsa_c128_cos_sin,
    const std::vector<std::vector<DSACacheInfo>>& caches_info,
    const std::vector<DSAGroupInfo>& group_infos,
    DSAMetadata& dsa) {
  const int32_t batch_size =
      static_cast<int32_t>(params.kv_seq_lens_vec.size());
  std::vector<int> q_lens_vec;
  q_lens_vec.reserve(batch_size);

  dsa.input_positions = positions;

  // Build per-batch sequence length metadata.
  build_seq_lengths(params, batch_size, dsa);
  if (static_cast<int32_t>(params.q_seq_lens_vec.size()) == batch_size) {
    q_lens_vec.assign(params.q_seq_lens_vec.begin(),
                      params.q_seq_lens_vec.end());
  } else if (params.batch_forward_type.no_decode()) {
    q_lens_vec.assign(params.kv_seq_lens_vec.begin(),
                      params.kv_seq_lens_vec.end());
  } else {
    q_lens_vec.assign(batch_size, 1);
  }

  // Keep base RoPE tables in metadata. Per-forward cos/sin slices are
  // calculated in DeepseekV4ModelImpl::forward to align with MindIE timing.
  if (dsa_cos_sin.defined()) {
    auto cos_sin_chunks = dsa_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    dsa.cos_table = cos_sin_chunks[0].contiguous();
    dsa.sin_table = cos_sin_chunks[1].contiguous();
  }

  if (positions.defined()) {
    const int64_t total_tokens = positions.numel();
    build_positions(params, batch_size, total_tokens, dsa);
  }

  (void)dsa_c4_cos_sin;
  (void)dsa_c128_cos_sin;

  // --- Block tables / slots expansion ---
  if (!params.multi_block_tables.empty() && !caches_info.empty()) {
    std::vector<torch::Tensor> active_multi_block_tables =
        params.multi_block_tables;
    int32_t manager_num =
        static_cast<int32_t>(active_multi_block_tables.size());
    if (manager_num == 1 && batch_size == 1 &&
        active_multi_block_tables[0].defined() &&
        active_multi_block_tables[0].dim() == 2 &&
        active_multi_block_tables[0].size(0) > 1 &&
        static_cast<size_t>(active_multi_block_tables[0].size(0)) <=
            group_infos.size()) {
      const auto packed = active_multi_block_tables[0].contiguous();
      std::vector<torch::Tensor> unpacked_tables;
      unpacked_tables.reserve(packed.size(0));
      for (int64_t m = 0; m < packed.size(0); ++m) {
        unpacked_tables.push_back(packed[m].unsqueeze(0).contiguous());
      }
      active_multi_block_tables = std::move(unpacked_tables);
      manager_num = static_cast<int32_t>(active_multi_block_tables.size());
      LOG(WARNING)
          << "DSAMetadataBuilder detected packed multi_block_tables layout "
             "([manager, blocks]) while batch_size==1; auto-unpacked to "
             "[manager][batch, blocks]. manager_num="
          << manager_num;
    }

    CHECK_EQ(batch_size, static_cast<int32_t>(params.kv_seq_lens_vec.size()))
        << "DSAMetadataBuilder: batch_size mismatch with kv_seq_lens_vec size.";
    CHECK_LE(manager_num, static_cast<int32_t>(group_infos.size()))
        << "DSAMetadataBuilder: manager_num(" << manager_num
        << ") exceeds group_infos size(" << group_infos.size()
        << "), cannot align manager/group mapping.";
    const int32_t n_layers = static_cast<int32_t>(caches_info.size());
    const auto& ctx_lens = params.kv_seq_lens_vec;
    int64_t total_tokens = 0;
    for (auto len : ctx_lens) total_tokens += len;

    // Step 1: block -> slot expansion per manager
    std::vector<torch::Tensor> mgr_slots(manager_num);
    for (int32_t m = 0; m < manager_num; ++m) {
      mgr_slots[m] = expand_blocks_to_slots(active_multi_block_tables[m],
                                            group_infos[m],
                                            ctx_lens,
                                            batch_size,
                                            total_tokens);
    }

    // Step 2: per-group processing
    std::vector<torch::Tensor> proc_slots(manager_num);
    std::vector<torch::Tensor> proc_bt(manager_num);
    for (int32_t m = 0; m < manager_num; ++m) {
      process_group(active_multi_block_tables[m],
                    mgr_slots[m],
                    group_infos[m],
                    ctx_lens,
                    q_lens_vec,
                    batch_size,
                    total_tokens,
                    proc_bt[m],
                    proc_slots[m]);
    }

    // Metadata expansion uses CPU accessor paths. Move processed tables/sl
    // slots back to model device once per forward for downstream NPU kernels.
    const torch::Device target_device =
        positions.defined() ? positions.device() : torch::Device(torch::kCPU);
    if (!target_device.is_cpu()) {
      for (int32_t m = 0; m < manager_num; ++m) {
        if (proc_bt[m].defined() && proc_bt[m].device() != target_device) {
          proc_bt[m] = safe_to(
              proc_bt[m], proc_bt[m].options().device(target_device), true);
        }
        if (proc_slots[m].defined() &&
            proc_slots[m].device() != target_device) {
          proc_slots[m] = safe_to(proc_slots[m],
                                  proc_slots[m].options().device(target_device),
                                  true);
        }
      }
    }

    // Step 3: expand by layer using group_id
    dsa.block_tables.resize(n_layers);
    dsa.slot_mappings.resize(n_layers);
    for (int32_t lid = 0; lid < n_layers; ++lid) {
      const auto& lci = caches_info[lid];
      dsa.block_tables[lid].resize(lci.size());
      dsa.slot_mappings[lid].resize(lci.size());
      for (size_t ci = 0; ci < lci.size(); ++ci) {
        int32_t gid = lci[ci].group_id;
        if (gid < manager_num) {
          dsa.block_tables[lid][ci] = proc_bt[gid];
          dsa.slot_mappings[lid][ci] = proc_slots[gid];
        }
      }
    }
  }

  // Attach cache spec pointer
  dsa.caches_info = &caches_info;
}

torch::Tensor DSAMetadataBuilder::expand_blocks_to_slots(
    const torch::Tensor& block_table,
    const DSAGroupInfo& gi,
    const std::vector<int>& ctx_lens,
    int32_t batch_size,
    int64_t total_tokens) {
  const int32_t bs = gi.block_size;
  auto slots = torch::full({total_tokens}, -1, torch::kInt32);
  auto slots_acc = slots.accessor<int32_t, 1>();
  auto bt_acc = block_table.accessor<int32_t, 2>();
  const int32_t max_blocks = block_table.size(1);

  int64_t start_idx = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    int64_t token_len = ctx_lens[seq];
    int64_t slot_num = compute_slot_num(gi, token_len);

    int64_t filled = 0;
    for (int32_t blk = 0; blk < max_blocks && filled < slot_num; ++blk) {
      int32_t block_id = bt_acc[seq][blk];
      if (block_id < 0) break;
      for (int32_t off = 0; off < bs && filled < slot_num; ++off) {
        slots_acc[start_idx + filled] =
            static_cast<int32_t>(static_cast<int64_t>(block_id) * bs + off);
        ++filled;
      }
    }
    start_idx += token_len;
  }
  return slots;
}

int64_t DSAMetadataBuilder::compute_slot_num(const DSAGroupInfo& gi,
                                             int64_t token_len) {
  if (gi.type == DSACacheType::TOKEN) {
    return token_len / gi.ratio;
  }
  // SLIDING_WINDOW
  const int32_t bs = gi.block_size;
  if (token_len > bs) {
    return token_len % bs + bs;
  }
  int64_t n = token_len % bs;
  return (n == 0 && token_len > 0) ? bs : n;
}

void DSAMetadataBuilder::process_group(const torch::Tensor& raw_bt,
                                       const torch::Tensor& raw_slots,
                                       const DSAGroupInfo& gi,
                                       const std::vector<int>& ctx_lens,
                                       const std::vector<int>& q_lens,
                                       int32_t batch_size,
                                       int64_t total_tokens,
                                       torch::Tensor& out_bt,
                                       torch::Tensor& out_slots) {
  if (gi.type == DSACacheType::TOKEN) {
    process_token_group(raw_bt,
                        raw_slots,
                        gi.ratio,
                        ctx_lens,
                        q_lens,
                        batch_size,
                        total_tokens,
                        out_bt,
                        out_slots);
  } else if (gi.type == DSACacheType::SLIDING_WINDOW) {
    process_swa_group(raw_bt,
                      raw_slots,
                      gi.block_size,
                      ctx_lens,
                      batch_size,
                      out_bt,
                      out_slots);
  } else {
    out_slots =
        torch::where(raw_slots.eq(-1), torch::zeros_like(raw_slots), raw_slots);
    out_bt = raw_bt;
  }
}

void DSAMetadataBuilder::process_token_group(const torch::Tensor& raw_bt,
                                             const torch::Tensor& raw_slots,
                                             int32_t ratio,
                                             const std::vector<int>& ctx_lens,
                                             const std::vector<int>& q_lens,
                                             int32_t batch_size,
                                             int64_t total_tokens,
                                             torch::Tensor& out_bt,
                                             torch::Tensor& out_slots) {
  CHECK_EQ(static_cast<int32_t>(ctx_lens.size()), batch_size)
      << "process_token_group requires ctx_lens.size == batch_size, got "
      << ctx_lens.size() << " vs " << batch_size;
  CHECK_EQ(static_cast<int32_t>(q_lens.size()), batch_size)
      << "process_token_group requires q_lens.size == batch_size, got "
      << q_lens.size() << " vs " << batch_size;

  int64_t query_total_tokens = 0;
  for (const int q_len : q_lens) {
    query_total_tokens += static_cast<int64_t>(q_len);
  }
  int64_t op_need_length =
      std::min(query_total_tokens / ratio + static_cast<int64_t>(batch_size),
               query_total_tokens);

  // Token caches write only the compressed rows produced by the current
  // forward step; trailing rows remain zero-padded as dummy slots.
  auto out_slots_tensor = torch::zeros({op_need_length}, raw_slots.options());
  auto out_slots_acc = out_slots_tensor.accessor<int32_t, 1>();
  auto raw_slots_acc = raw_slots.accessor<int32_t, 1>();

  int64_t start_idx = 0;
  int64_t write_idx = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    const int64_t ctx_len = static_cast<int64_t>(ctx_lens[seq]);
    const int64_t q_len =
        std::clamp<int64_t>(static_cast<int64_t>(q_lens[seq]), 0, ctx_len);
    const int64_t prev_ctx_len = ctx_len - q_len;
    const int64_t prev_committed = prev_ctx_len / ratio;
    const int64_t committed = ctx_len / ratio;
    const int64_t new_committed = committed - prev_committed;
    for (int64_t i = 0; i < new_committed && write_idx < op_need_length; ++i) {
      out_slots_acc[write_idx++] =
          raw_slots_acc[start_idx + prev_committed + i];
    }
    start_idx += ctx_len;
  }

  CHECK_LE(write_idx, op_need_length)
      << "process_token_group generated more committed slots than expected: "
      << "write_idx=" << write_idx << ", op_need_length=" << op_need_length
      << ", query_total_tokens=" << query_total_tokens << ", ratio=" << ratio
      << ", batch_size=" << batch_size << ", total_tokens=" << total_tokens;
  out_slots = out_slots_tensor;
  out_bt = raw_bt;  // keep original right-padded block_tables
}

void DSAMetadataBuilder::process_swa_group(const torch::Tensor& raw_bt,
                                           const torch::Tensor& raw_slots,
                                           int32_t block_size,
                                           const std::vector<int>& ctx_lens,
                                           int32_t batch_size,
                                           torch::Tensor& out_bt,
                                           torch::Tensor& out_slots) {
  out_slots =
      torch::where(raw_slots.eq(-1), torch::zeros_like(raw_slots), raw_slots);

  int32_t current_cols = raw_bt.size(1);
  int32_t max_dst_len = 0;
  std::vector<int32_t> dst_lens(batch_size);
  for (int32_t s = 0; s < batch_size; ++s) {
    dst_lens[s] = static_cast<int32_t>(
        std::ceil(static_cast<double>(ctx_lens[s]) / block_size));
    max_dst_len = std::max(max_dst_len, dst_lens[s]);
  }
  max_dst_len = std::max(max_dst_len, current_cols);

  auto new_bt = torch::zeros({batch_size, max_dst_len}, raw_bt.options());
  auto new_acc = new_bt.accessor<int32_t, 2>();
  auto old_acc = raw_bt.accessor<int32_t, 2>();

  for (int32_t s = 0; s < batch_size; ++s) {
    int32_t pad_len = dst_lens[s] - current_cols;
    if (pad_len > 0) {
      for (int32_t j = 0; j < current_cols; ++j)
        new_acc[s][pad_len + j] = old_acc[s][j];
    } else if (pad_len < 0) {
      for (int32_t j = 0; j < dst_lens[s]; ++j) new_acc[s][j] = old_acc[s][j];
    } else {
      for (int32_t j = 0; j < current_cols; ++j) new_acc[s][j] = old_acc[s][j];
    }
  }
  out_bt = new_bt;
}

void DSAMetadataBuilder::build_seq_lengths(const ModelInputParams& params,
                                           int32_t batch_size,
                                           DSAMetadata& dsa_metadata) {
  auto kv_lens =
      torch::tensor(std::vector<int32_t>(params.kv_seq_lens_vec.begin(),
                                         params.kv_seq_lens_vec.end()),
                    torch::kInt32);
  dsa_metadata.seq_lens = kv_lens;
  dsa_metadata.actual_seq_lengths_kv = kv_lens;

  torch::Tensor q_lens;
  if (static_cast<int32_t>(params.q_seq_lens_vec.size()) == batch_size) {
    // Prefer explicit per-sequence query lengths from ModelInputParams.
    // This is accurate for prefill/decode/chunked/mixed batches.
    q_lens = torch::tensor(std::vector<int32_t>(params.q_seq_lens_vec.begin(),
                                                params.q_seq_lens_vec.end()),
                           torch::kInt32);
  } else if (params.batch_forward_type.no_decode()) {
    // Pure prefill path fallback: query lengths follow KV context lengths.
    q_lens = kv_lens;
  } else {
    // Decode fallback: each sequence contributes one query token.
    q_lens = torch::ones({batch_size}, torch::kInt32);
  }
  // cumsum with leading 0: shape (batch_size+1,)
  auto cumsum = torch::cumsum(q_lens, /*dim=*/0, /*dtype=*/torch::kInt32);
  dsa_metadata.actual_seq_lengths_query =
      torch::cat({torch::zeros({1}, torch::kInt32), cumsum});
  dsa_metadata.seq_lens_q = q_lens;

  auto int_options = torch::TensorOptions().dtype(torch::kInt32);
  if (kv_lens.numel() > 0) {
    dsa_metadata.max_seqlen_kv = torch::max(kv_lens).to(torch::kInt32);
  } else {
    dsa_metadata.max_seqlen_kv = torch::zeros({1}, int_options);
  }

  if (q_lens.numel() > 0) {
    dsa_metadata.max_seqlen_q = torch::max(q_lens).to(torch::kInt32);
  } else {
    dsa_metadata.max_seqlen_q = torch::zeros({1}, int_options);
  }
}

void DSAMetadataBuilder::build_positions(const ModelInputParams& params,
                                         int32_t batch_size,
                                         int64_t total_tokens,
                                         DSAMetadata& dsa_metadata) {
  (void)params;
  (void)total_tokens;
  if (!dsa_metadata.input_positions.defined()) return;

  auto input_positions = dsa_metadata.input_positions;
  int64_t num_tokens = input_positions.size(0);

  // C4 compressed positions
  auto c4_mask = ((input_positions + 1) % 4).eq(0);
  auto c4_pos = input_positions.index({c4_mask});
  c4_pos = ((c4_pos + 1) / 4) - 1;
  int64_t c4_target = std::min(num_tokens, num_tokens / 4 + batch_size);
  int64_t c4_pad_right = c4_target - c4_pos.size(0);
  if (c4_pad_right > 0) {
    dsa_metadata.c4_pad_positions =
        torch::cat({c4_pos, torch::zeros({c4_pad_right}, c4_pos.options())});
  } else {
    dsa_metadata.c4_pad_positions = c4_pos.slice(0, 0, c4_target);
  }

  // C128 compressed positions
  auto c128_mask = ((input_positions + 1) % 128).eq(0);
  auto c128_pos = input_positions.index({c128_mask});
  c128_pos = ((c128_pos + 1) / 128) - 1;
  int64_t c128_target = std::min(num_tokens, num_tokens / 128 + batch_size);
  int64_t c128_pad_right = c128_target - c128_pos.size(0);
  if (c128_pad_right > 0) {
    dsa_metadata.c128_pad_positions = torch::cat(
        {c128_pos, torch::zeros({c128_pad_right}, c128_pos.options())});
  } else {
    dsa_metadata.c128_pad_positions = c128_pos.slice(0, 0, c128_target);
  }
}

void DSAMetadataBuilder::build_group_cos_sin(const torch::Tensor& cos_sin_table,
                                             const torch::Tensor& pad_positions,
                                             torch::Tensor& out_cos,
                                             torch::Tensor& out_sin) {
  if (!cos_sin_table.defined() || !pad_positions.defined() ||
      pad_positions.numel() == 0) {
    return;
  }

  auto group_positions = pad_positions;
  if (group_positions.scalar_type() != torch::kInt64) {
    group_positions = group_positions.to(torch::kInt64);
  }

  auto group_table = cos_sin_table;
  if (group_table.device() != group_positions.device()) {
    group_table = group_table.to(group_positions.device());
  }

  auto target = group_table.index({group_positions});
  auto chunks = target.chunk(/*chunks=*/2, /*dim=*/-1);
  out_cos = chunks[0].contiguous();
  out_sin = chunks[1].contiguous();
}

}  // namespace xllm::layer
