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

#include "runtime/params_utils.h"

#include <absl/strings/str_join.h>
#include <torch/torch.h>

#include <optional>

#include "common/macros.h"
#include "common/metrics.h"
#include "framework/model/model_input_params.h"
#include "runtime/forward_params.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
namespace {
template <typename T>
void aprint(std::vector<T> v, const std::string& name, int global_rank) {
  std::string value = absl::StrJoin(
      v, ", ", [](std::string* out, T n) { out->append(std::to_string(n)); });
  LOG(INFO) << "GlobalRank = " << global_rank << ", name = " << name
            << ", value = " << value;
}
}  // namespace

void proto_to_forward_input(const proto::ForwardInput* pb_forward_input,
                            ForwardInput& forward_inputs,
                            int64_t num_decoding_tokens) {
  Timer timer;
  int32_t num_sequences = pb_forward_input->num_sequences();
  std::vector<int32_t> flatten_tokens_vec =
      std::vector<int32_t>(pb_forward_input->flatten_tokens_vec().begin(),
                           pb_forward_input->flatten_tokens_vec().end());
  std::vector<int32_t> flatten_positions_vec =
      std::vector<int32_t>(pb_forward_input->flatten_positions_vec().begin(),
                           pb_forward_input->flatten_positions_vec().end());
  std::vector<int32_t> new_token_slot_ids =
      std::vector<int32_t>(pb_forward_input->new_token_slot_ids().begin(),
                           pb_forward_input->new_token_slot_ids().end());
  // aprint<int32_t>(new_token_slot_ids, "token_slot_ids", global_rank_);
  std::vector<int32_t> seq_lens = std::vector<int32_t>(
      pb_forward_input->seq_lens().begin(), pb_forward_input->seq_lens().end());
  // aprint<int32_t>(seq_lens, "seq_lens", global_rank_);
  std::vector<int32_t> q_seq_lens =
      std::vector<int32_t>(pb_forward_input->q_seq_lens().begin(),
                           pb_forward_input->q_seq_lens().end());
  // aprint<int32_t>(q_seq_lens, "q_seq_lens", global_rank_);
  std::vector<std::vector<int32_t>> block_tables_vec;
  for (size_t i = 0; i < pb_forward_input->block_tables_vec().size(); ++i) {
    block_tables_vec.emplace_back(std::vector<int32_t>(
        pb_forward_input->block_tables_vec()[i].block_tables().begin(),
        pb_forward_input->block_tables_vec()[i].block_tables().end()));
    // aprint<int32_t>((block_tables_vec.back()), "block_tables_vec",
    // global_rank_);
  }
  std::vector<int32_t> selected_token_idxes =
      std::vector<int32_t>(pb_forward_input->selected_token_idxes().begin(),
                           pb_forward_input->selected_token_idxes().end());
  // aprint<int32_t>(selected_token_idxes, "selected_token_idxes",
  // global_rank_);
  std::vector<int32_t> sample_idxes =
      std::vector<int32_t>(pb_forward_input->sample_idxes().begin(),
                           pb_forward_input->sample_idxes().end());
  // aprint<int32_t>(sample_idxes, "sample_idxes", global_rank_);

  std::vector<std::vector<int64_t>> unique_token_ids_vec;
  for (size_t i = 0; i < pb_forward_input->unique_token_ids_vec().size(); ++i) {
    unique_token_ids_vec.emplace_back(std::vector<int64_t>(
        pb_forward_input->unique_token_ids_vec()[i].unique_token_ids().begin(),
        pb_forward_input->unique_token_ids_vec()[i].unique_token_ids().end()));
    // aprint<int32_t>((unique_token_ids_vec.back()), "unique_token_ids_vec",
    // global_rank_);
  }
  std::vector<std::vector<int32_t>> unique_token_counts_vec;
  for (size_t i = 0; i < pb_forward_input->unique_token_counts_vec().size();
       ++i) {
    unique_token_counts_vec.emplace_back(
        std::vector<int32_t>(pb_forward_input->unique_token_counts_vec()[i]
                                 .unique_token_counts()
                                 .begin(),
                             pb_forward_input->unique_token_counts_vec()[i]
                                 .unique_token_counts()
                                 .end()));
    // aprint<int32_t>((unique_token_counts_vec.back()),
    // "unique_token_counts_vec", global_rank_);
  }
  std::vector<int32_t> unique_token_lens_vec =
      std::vector<int32_t>(pb_forward_input->unique_token_lens_vec().begin(),
                           pb_forward_input->unique_token_lens_vec().end());
  // aprint<int32_t>(unique_token_lens_vec, "unique_token_lens_vec",
  // global_rank_);

  std::vector<int32_t> embedding_ids =
      std::vector<int32_t>(pb_forward_input->embedding_ids().begin(),
                           pb_forward_input->embedding_ids().end());

  std::vector<const RequestSamplingParam*> sampling_params;
  std::vector<RequestSamplingParam> tmp_sampling_params;
  for (auto sp : pb_forward_input->sampling_params()) {
    RequestSamplingParam tmp;
    tmp.frequency_penalty = sp.frequency_penalty();
    tmp.presence_penalty = sp.presence_penalty();
    tmp.repetition_penalty = sp.repetition_penalty();
    tmp.temperature = sp.temperature();
    tmp.top_p = sp.top_p();
    tmp.top_k = sp.top_k();
    tmp.logprobs = sp.logprobs();
    tmp.top_logprobs = sp.top_logprobs();
    tmp.do_sample = sp.do_sample();
    tmp.is_embeddings = sp.is_embeddings();
    tmp_sampling_params.emplace_back(tmp);
  }
  for (size_t i = 0; i < tmp_sampling_params.size(); ++i) {
    sampling_params.emplace_back(&tmp_sampling_params[i]);
  }

  std::vector<int32_t> dp_global_token_nums =
      std::vector<int32_t>(pb_forward_input->dp_global_token_nums().begin(),
                           pb_forward_input->dp_global_token_nums().end());

  // Create ForwardInput on cpu pinned memory here
  auto tensor_options = torch::TensorOptions()
                            .dtype(torch::kInt)
                            .device(torch::kCPU)
                            .pinned_memory(true);
  forward_inputs.token_ids = torch::tensor(flatten_tokens_vec, tensor_options);
  forward_inputs.positions =
      torch::tensor(flatten_positions_vec, tensor_options);
  std::pair<int, int> prefill_indices{0, 0};
#if defined(USE_NPU)
  if (q_seq_lens.size() >= 1) {
    prefill_indices = util::find_ones_indices(q_seq_lens);
  }
#endif
  auto& input_params = forward_inputs.input_params;
  input_params.empty_kv_cache = pb_forward_input->empty_kv_cache();
  input_params.global_empty_kv_cache =
      pb_forward_input->global_empty_kv_cache();
  input_params.num_sequences = block_tables_vec.size();
  input_params.kv_max_seq_len = pb_forward_input->max_seq_len();
  input_params.q_max_seq_len = pb_forward_input->q_max_seq_len();
  input_params.kv_seq_lens = torch::tensor(seq_lens, tensor_options);
  input_params.q_seq_lens = torch::tensor(q_seq_lens, tensor_options);
  input_params.kv_seq_lens_vec = std::move(seq_lens);
  input_params.q_seq_lens_vec = std::move(q_seq_lens);

  input_params.new_cache_slots =
      torch::tensor(new_token_slot_ids, tensor_options);
  input_params.prefill_indices = prefill_indices;

  util::pad_2d_vector(block_tables_vec, /*pad_value=*/0);
  input_params.block_tables =
      std::move(create_2d_tensor(block_tables_vec, torch::kInt));

  input_params.dp_global_token_nums = std::move(dp_global_token_nums);
  input_params.embedding_ids = std::move(embedding_ids);

  if (pb_forward_input->embeds().size() > 0) {
    const int32_t rows = pb_forward_input->embeds().size();
    const int32_t cols = pb_forward_input->embeds()[0].vals().size();
    std::vector<std::vector<float>> embeddings_vec;
    for (size_t i = 0; i < rows; ++i) {
      embeddings_vec.emplace_back(
          std::vector<float>(pb_forward_input->embeds()[i].vals().begin(),
                             pb_forward_input->embeds()[i].vals().end()));
    }
    torch::Tensor embeddings =
        create_2d_tensor(embeddings_vec, torch::kBFloat16);
    input_params.mm_data =
        MMData(MMType::EMBEDDING, {{"embedding", embeddings}});
  }

  CHECK_EQ(sampling_params.size(), selected_token_idxes.size());
  if (!selected_token_idxes.empty()) {
    util::pad_2d_vector<int64_t>(unique_token_ids_vec, /*pad_value=*/0);
    util::pad_2d_vector(unique_token_counts_vec, /*pad_value=*/0);
    forward_inputs.sampling_params.init(sampling_params,
                                        selected_token_idxes,
                                        sample_idxes,
                                        unique_token_ids_vec,
                                        unique_token_counts_vec,
                                        unique_token_lens_vec);
  }

  forward_inputs.transfer_kv_infos.reserve(
      pb_forward_input->transfer_kv_infos().size());
  for (int i = 0; i < pb_forward_input->transfer_kv_infos().size(); ++i) {
    TransferKVInfo transfer_kv_info;
    transfer_kv_info.request_id =
        pb_forward_input->transfer_kv_infos()[i].request_id();
    transfer_kv_info.local_blocks_ids = std::vector<uint64_t>(
        pb_forward_input->transfer_kv_infos()[i].local_blocks_ids().begin(),
        pb_forward_input->transfer_kv_infos()[i].local_blocks_ids().end());
    transfer_kv_info.remote_blocks_ids = std::vector<uint64_t>(
        pb_forward_input->transfer_kv_infos()[i].remote_blocks_ids().begin(),
        pb_forward_input->transfer_kv_infos()[i].remote_blocks_ids().end());
    transfer_kv_info.dp_rank =
        pb_forward_input->transfer_kv_infos()[i].dp_rank();

    InstanceInfo instance_info;
    instance_info.name =
        pb_forward_input->transfer_kv_infos()[i].remote_instance_info().name();
    instance_info.rpc_address = pb_forward_input->transfer_kv_infos()[i]
                                    .remote_instance_info()
                                    .rpc_address();
    instance_info.type =
        pb_forward_input->transfer_kv_infos()[i].remote_instance_info().type();
    instance_info.cluster_ids =
        std::vector<uint64_t>(pb_forward_input->transfer_kv_infos()[i]
                                  .remote_instance_info()
                                  .cluster_ids()
                                  .begin(),
                              pb_forward_input->transfer_kv_infos()[i]
                                  .remote_instance_info()
                                  .cluster_ids()
                                  .end());
    instance_info.addrs =
        std::vector<std::string>(pb_forward_input->transfer_kv_infos()[i]
                                     .remote_instance_info()
                                     .addrs()
                                     .begin(),
                                 pb_forward_input->transfer_kv_infos()[i]
                                     .remote_instance_info()
                                     .addrs()
                                     .end());
    instance_info.k_cache_ids =
        std::vector<int64_t>(pb_forward_input->transfer_kv_infos()[i]
                                 .remote_instance_info()
                                 .k_cache_ids()
                                 .begin(),
                             pb_forward_input->transfer_kv_infos()[i]
                                 .remote_instance_info()
                                 .k_cache_ids()
                                 .end());
    instance_info.v_cache_ids =
        std::vector<int64_t>(pb_forward_input->transfer_kv_infos()[i]
                                 .remote_instance_info()
                                 .v_cache_ids()
                                 .begin(),
                             pb_forward_input->transfer_kv_infos()[i]
                                 .remote_instance_info()
                                 .v_cache_ids()
                                 .end());
    instance_info.dp_size = pb_forward_input->transfer_kv_infos()[i]
                                .remote_instance_info()
                                .dp_size();

    transfer_kv_info.remote_instance_info = std::move(instance_info);
    forward_inputs.transfer_kv_infos.emplace_back(std::move(transfer_kv_info));
  }
  COUNTER_ADD(proto_latency_seconds_proto2i, timer.elapsed_seconds());
}

void forward_input_to_proto(const RawForwardInput& inputs,
                            proto::ForwardInput* pb_forward_input) {
  Timer timer;
  ADD_VECTOR_TO_PROTO(pb_forward_input->mutable_flatten_tokens_vec(),
                      inputs.flatten_tokens_vec);
  ADD_VECTOR_TO_PROTO(pb_forward_input->mutable_flatten_positions_vec(),
                      inputs.flatten_positions_vec);
  std::vector<proto::RequestSamplingParam> pb_sampling_params;
  for (auto sp : inputs.sampling_params) {
    proto::RequestSamplingParam pb_sp;
    pb_sp.set_frequency_penalty(sp->frequency_penalty);
    pb_sp.set_presence_penalty(sp->presence_penalty);
    pb_sp.set_repetition_penalty(sp->repetition_penalty);
    pb_sp.set_temperature(sp->temperature);
    pb_sp.set_top_p(sp->top_p);
    pb_sp.set_top_k(sp->top_k);
    pb_sp.set_logprobs(sp->logprobs);
    pb_sp.set_top_logprobs(sp->top_logprobs);
    pb_sp.set_do_sample(sp->do_sample);
    pb_sp.set_is_embeddings(sp->is_embeddings);
    pb_sampling_params.emplace_back(pb_sp);
  }
  ADD_VECTOR_TO_PROTO(pb_forward_input->mutable_sampling_params(),
                      pb_sampling_params);
  ADD_VECTOR_TO_PROTO(pb_forward_input->mutable_selected_token_idxes(),
                      inputs.selected_token_idxes);
  ADD_VECTOR_TO_PROTO(pb_forward_input->mutable_sample_idxes(),
                      inputs.sample_idxes);
  pb_forward_input->mutable_unique_token_ids_vec()->Reserve(
      inputs.unique_token_ids_vec.size());
  for (auto ids : inputs.unique_token_ids_vec) {
    proto::UniqueTokenIds pb_unique_token_ids;
    ADD_VECTOR_TO_PROTO(pb_unique_token_ids.mutable_unique_token_ids(), ids);
    *pb_forward_input->mutable_unique_token_ids_vec()->Add() =
        pb_unique_token_ids;
  }
  pb_forward_input->mutable_unique_token_counts_vec()->Reserve(
      inputs.unique_token_counts_vec.size());
  for (auto counts : inputs.unique_token_counts_vec) {
    proto::UniqueTokenCounts pb_unique_token_counts;
    ADD_VECTOR_TO_PROTO(pb_unique_token_counts.mutable_unique_token_counts(),
                        counts);
    *pb_forward_input->mutable_unique_token_counts_vec()->Add() =
        pb_unique_token_counts;
  }
  ADD_VECTOR_TO_PROTO(pb_forward_input->mutable_unique_token_lens_vec(),
                      inputs.unique_token_lens_vec);
  pb_forward_input->set_empty_kv_cache(inputs.empty_kv_cache);
  pb_forward_input->set_global_empty_kv_cache(inputs.global_empty_kv_cache);
  pb_forward_input->set_max_seq_len(inputs.max_seq_len);
  pb_forward_input->set_q_max_seq_len(inputs.q_max_seq_len);
  ADD_VECTOR_TO_PROTO(pb_forward_input->mutable_seq_lens(), inputs.seq_lens);
  ADD_VECTOR_TO_PROTO(pb_forward_input->mutable_q_seq_lens(),
                      inputs.q_seq_lens);
  ADD_VECTOR_TO_PROTO(pb_forward_input->mutable_new_token_slot_ids(),
                      inputs.new_token_slot_ids);
  pb_forward_input->mutable_block_tables_vec()->Reserve(
      inputs.block_tables_vec.size());
  for (auto t : inputs.block_tables_vec) {
    proto::BlockTables pb_table;
    ADD_VECTOR_TO_PROTO(pb_table.mutable_block_tables(), t);
    *pb_forward_input->mutable_block_tables_vec()->Add() = pb_table;
  }
  pb_forward_input->set_num_sequences(inputs.num_sequences);
  ADD_VECTOR_TO_PROTO(pb_forward_input->mutable_dp_global_token_nums(),
                      inputs.dp_global_token_nums);
  if (!inputs.transfer_kv_infos.empty()) {
    pb_forward_input->mutable_transfer_kv_infos()->Reserve(
        inputs.transfer_kv_infos.size());
    for (auto& transfer_kv_info : inputs.transfer_kv_infos) {
      auto pb_transfer_kv_info =
          pb_forward_input->mutable_transfer_kv_infos()->Add();
      pb_transfer_kv_info->set_request_id(transfer_kv_info.request_id);
      ADD_VECTOR_TO_PROTO(pb_transfer_kv_info->mutable_local_blocks_ids(),
                          transfer_kv_info.local_blocks_ids);
      ADD_VECTOR_TO_PROTO(pb_transfer_kv_info->mutable_remote_blocks_ids(),
                          transfer_kv_info.remote_blocks_ids);
      pb_transfer_kv_info->set_dp_rank(transfer_kv_info.dp_rank);
      pb_transfer_kv_info->mutable_remote_instance_info()->set_name(
          transfer_kv_info.remote_instance_info.name);
      pb_transfer_kv_info->mutable_remote_instance_info()->set_rpc_address(
          transfer_kv_info.remote_instance_info.rpc_address);
      pb_transfer_kv_info->mutable_remote_instance_info()->set_type(
          transfer_kv_info.remote_instance_info.type);
      ADD_VECTOR_TO_PROTO(pb_transfer_kv_info->mutable_remote_instance_info()
                              ->mutable_cluster_ids(),
                          transfer_kv_info.remote_instance_info.cluster_ids);
      ADD_VECTOR_TO_PROTO(
          pb_transfer_kv_info->mutable_remote_instance_info()->mutable_addrs(),
          transfer_kv_info.remote_instance_info.addrs);
      ADD_VECTOR_TO_PROTO(pb_transfer_kv_info->mutable_remote_instance_info()
                              ->mutable_k_cache_ids(),
                          transfer_kv_info.remote_instance_info.k_cache_ids);
      ADD_VECTOR_TO_PROTO(pb_transfer_kv_info->mutable_remote_instance_info()
                              ->mutable_v_cache_ids(),
                          transfer_kv_info.remote_instance_info.v_cache_ids);
      pb_transfer_kv_info->mutable_remote_instance_info()->set_dp_size(
          transfer_kv_info.remote_instance_info.dp_size);
    }
  }
  pb_forward_input->mutable_embeds()->Reserve(inputs.embeddings.size());
  for (auto t : inputs.embeddings) {
    proto::Embeddings embeds;
    ADD_VECTOR_TO_PROTO(embeds.mutable_vals(), t);
    *pb_forward_input->mutable_embeds()->Add() = embeds;
  }
  COUNTER_ADD(proto_latency_seconds_i2proto, timer.elapsed_seconds());
  pb_forward_input->set_prefill_seq_len(inputs.prefill_seq_len);
  ADD_VECTOR_TO_PROTO(pb_forward_input->mutable_embedding_ids(),
                      inputs.embedding_ids);
}

void proto_to_forward_output(const proto::ForwardOutput& pb_output,
                             RawForwardOutput& raw_forward_output) {
  Timer timer;
  size_t seq_nums = pb_output.outputs().size();
  raw_forward_output.outputs.reserve(seq_nums);
  for (size_t i = 0; i < seq_nums; ++i) {
    proto::SquenceOutput pb_seq_out = pb_output.outputs()[i];
    RawSampleOutput s;
    size_t token_nums = pb_seq_out.tokens().size();
    s.tokens.reserve(token_nums);
    for (size_t j = 0; j < token_nums; ++j) {
      RawToken t;
      t.id = pb_seq_out.tokens()[j].id();
      switch (pb_seq_out.tokens()[j].lp_case()) {
        case proto::Token::kEmpty:
          break;
        case proto::Token::kLogprob:
          t.logprob = pb_seq_out.tokens()[j].logprob();
          break;
        default:
          break;
      }
      t.top_tokens.assign(pb_seq_out.tokens()[j].top_tokens().begin(),
                          pb_seq_out.tokens()[j].top_tokens().end());
      t.top_logprobs.assign(pb_seq_out.tokens()[j].top_logprobs().begin(),
                            pb_seq_out.tokens()[j].top_logprobs().end());
      t.embeddings.assign(pb_seq_out.tokens()[j].embeddings().vals().begin(),
                          pb_seq_out.tokens()[j].embeddings().vals().end());
      s.tokens.emplace_back(t);
    }
    raw_forward_output.outputs.emplace_back(s);
  }
  COUNTER_ADD(proto_latency_seconds_proto2o, timer.elapsed_seconds());
}

void forward_output_to_proto(const torch::Tensor& next_tokens,
                             const torch::Tensor& logprobs,
                             const torch::Tensor& top_tokens,
                             const torch::Tensor& top_logprobs,
                             const torch::Tensor& embeddings,
                             proto::ForwardOutput* pb_forward_output) {
  Timer timer;
  const int32_t num_seqs = next_tokens.size(0);
  int32_t output_idx = 0;
  pb_forward_output->mutable_outputs()->Reserve(num_seqs);
  for (int32_t output_idx = 0; output_idx < num_seqs; ++output_idx) {
    if (next_tokens.dim() == 2) {
      const auto curr_idx = output_idx;
      const auto curr_next_tokens = next_tokens[curr_idx];
      const auto curr_logprobs =
          logprobs.defined() ? logprobs[curr_idx] : logprobs;
      const auto curr_top_tokens =
          top_tokens.defined() ? top_tokens[curr_idx] : top_tokens;
      const auto curr_top_logprobs =
          top_logprobs.defined() ? top_logprobs[curr_idx] : top_logprobs;
      const auto curr_embeddings =
          embeddings.defined() ? embeddings[curr_idx] : embeddings;

      int32_t num_tokens = curr_next_tokens.size(0);
      std::vector<Token> tokens;
      tokens.reserve(num_tokens);
      for (int32_t i = 0; i < num_tokens; ++i) {
        const auto token = build_token(i,
                                       curr_next_tokens,
                                       curr_logprobs,
                                       curr_top_tokens,
                                       curr_top_logprobs);
        if (token.id == -1) {
          break;
        }
        tokens.push_back(token);
      }
      num_tokens = tokens.size();
      proto::SquenceOutput pb_seq_out;
      pb_seq_out.mutable_tokens()->Reserve(num_tokens);
      for (int32_t i = 0; i < num_tokens; ++i) {
        const auto& token = tokens[i];
        proto::Token pb_token;
        pb_token.set_id(token.id);
        if (token.logprob.has_value()) {
          pb_token.set_logprob(token.logprob.value());
        } else {
          pb_token.set_empty(true);
        }
        pb_token.mutable_top_tokens()->Reserve(token.top_tokens.size());
        for (auto it = token.top_tokens.begin(); it != token.top_tokens.end();
             ++it) {
          pb_token.add_top_tokens(*it);
        }
        pb_token.mutable_top_logprobs()->Reserve(token.top_logprobs.size());
        for (auto it = token.top_logprobs.begin();
             it != token.top_logprobs.end();
             ++it) {
          pb_token.add_top_logprobs(*it);
        }
        const auto token_embeddings =
            curr_embeddings.defined() ? curr_embeddings[i] : curr_embeddings;
        if (token_embeddings.defined()) {
          Slice<float> embedding_slice = {token_embeddings.data_ptr<float>(),
                                          token_embeddings.size(0)};
          ADD_VECTOR_TO_PROTO(pb_token.mutable_embeddings()->mutable_vals(),
                              embedding_slice);
        }
        *pb_seq_out.mutable_tokens()->Add() = pb_token;
      }
      *pb_forward_output->mutable_outputs()->Add() = pb_seq_out;
    } else {
      const auto token = build_token(
          output_idx, next_tokens, logprobs, top_tokens, top_logprobs);
      proto::SquenceOutput pb_seq_out;
      pb_seq_out.mutable_tokens()->Reserve(1);
      proto::Token pb_token;
      pb_token.set_id(token.id);
      if (token.logprob.has_value()) {
        pb_token.set_logprob(token.logprob.value());
      } else {
        pb_token.set_empty(true);
      }
      pb_token.mutable_top_tokens()->Reserve(token.top_tokens.size());
      for (auto it = token.top_tokens.begin(); it != token.top_tokens.end();
           ++it) {
        pb_token.add_top_tokens(*it);
      }
      pb_token.mutable_top_logprobs()->Reserve(token.top_logprobs.size());
      for (auto it = token.top_logprobs.begin(); it != token.top_logprobs.end();
           ++it) {
        pb_token.add_top_logprobs(*it);
      }
      const auto token_embeddings =
          embeddings.defined() ? embeddings[output_idx] : embeddings;
      if (token_embeddings.defined()) {
        Slice<float> embedding_slice = {token_embeddings.data_ptr<float>(),
                                        token_embeddings.size(0)};
        ADD_VECTOR_TO_PROTO(pb_token.mutable_embeddings()->mutable_vals(),
                            embedding_slice);
      }
      *pb_seq_out.mutable_tokens()->Add() = pb_token;
      *pb_forward_output->mutable_outputs()->Add() = pb_seq_out;
    }
  }
  COUNTER_ADD(proto_latency_seconds_o2proto, timer.elapsed_seconds());
  return;
}

Token build_token(int64_t index,
                  torch::Tensor token_ids,
                  torch::Tensor logprobs,
                  torch::Tensor top_tokens,
                  torch::Tensor top_logprobs) {
  Token token(token_ids[index].item<int64_t>());
  if (logprobs.defined()) {
    token.logprob = logprobs[index].item<float>();
    if (top_tokens.defined() && top_logprobs.defined()) {
      auto topk_tokens = top_tokens[index];
      auto topk_logprobs = top_logprobs[index];
      const size_t size = topk_tokens.numel();
      token.top_tokens = {topk_tokens.const_data_ptr<int64_t>(), size};
      token.top_logprobs = {topk_logprobs.const_data_ptr<float>(), size};
    }
  }
  return token;
}

}  // namespace xllm
