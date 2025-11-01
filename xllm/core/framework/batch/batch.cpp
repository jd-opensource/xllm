/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "batch.h"

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <vector>

#include "batch_input_builder.h"
#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/batch/mposition.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "rec_batch_input_builder.h"
#include "runtime/params_utils.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {

Batch::Batch(Sequence* sequence) { add(sequence); }
Batch::Batch(const std::vector<Sequence*>& sequences) { add(sequences); }

void Batch::add(Sequence* sequence, uint32_t allowed_max_token) {
  CHECK(sequence != nullptr);
  CHECK(!sequence->finished());
  CHECK_GT(allowed_max_token, 0);

  sequences_.push_back(sequence);
  allowed_max_tokens_.push_back(allowed_max_token);

  const auto& input_embedding = sequence->get_input_embedding();
  if (input_embedding.defined())
    input_embeddings_vec_.emplace_back(input_embedding);

  const auto& mm_data = sequence->get_mm_data();
  // if (sequence->is_prefill_stage() &&  mm_data.valid()) // TODO:Compatible
  // With Chunked Prefill
  if ((sequence->kv_state().kv_cache_tokens_num() <
       sequence->num_prompt_tokens()) &&
      mm_data.valid())
    mm_data_vec_.emplace_back(mm_data);
}

void Batch::add(const std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    add(sequence);
  }
}

ForwardInput Batch::prepare_forward_input(uint32_t num_decoding_tokens,
                                          uint32_t min_decoding_batch_size,
                                          const ModelArgs& args) {
  if (FLAGS_backend == "rec") {
    return prepare_rec_forward_input(
        num_decoding_tokens, min_decoding_batch_size, args);
  }
  BatchInputBuilder builder(sequences_,
                            allowed_max_tokens_,
                            input_embeddings_vec_,
                            mm_data_vec_,
                            copy_in_cache_block_infos_,
                            copy_out_cache_block_infos_,
                            swap_cache_block_infos_,
                            &args);
  return builder.build_forward_input(num_decoding_tokens,
                                     min_decoding_batch_size);
}

ForwardInput Batch::prepare_rec_forward_input(uint32_t num_decoding_tokens,
                                              uint32_t min_decoding_batch_size,
                                              const ModelArgs& args,
                                              ThreadPool* thread_pool) {
  // Convert SequencesGroup* to std::unique_ptr<SequencesGroup> for
  // compatibility
  std::vector<std::unique_ptr<SequencesGroup>> sequence_groups_ptrs;
  for (auto* group : sequence_groups_) {
    // Note: This is a temporary workaround. In production, we should avoid this
    // conversion and modify the interface to work with raw pointers directly.
    sequence_groups_ptrs.emplace_back(std::unique_ptr<SequencesGroup>(group));
  }

  RecBatchInputBuilder builder(
      sequence_groups_ptrs,
      allowed_max_tokens_,
      input_embeddings_vec_,
      mm_data_vec_,
      copy_in_cache_block_infos_,
      copy_out_cache_block_infos_,
      swap_cache_block_infos_,
      &args,
      thread_pool);  // Temporarily not using thread pool

  auto result = builder.build_rec_forward_input(num_decoding_tokens,
                                                min_decoding_batch_size);

  // Release the unique_ptrs without deleting the objects
  for (auto& ptr : sequence_groups_ptrs) {
    ptr.release();
  }

  return result;
}

RawForwardInput Batch::prepare_forward_input(uint32_t start_idx,
                                             uint32_t end_idx,
                                             ThreadPool* thread_pool) {
  BatchInputBuilder builder(sequences_,
                            allowed_max_tokens_,
                            input_embeddings_vec_,
                            mm_data_vec_,
                            copy_in_cache_block_infos_,
                            copy_out_cache_block_infos_,
                            swap_cache_block_infos_,
                            nullptr,
                            thread_pool);
  return builder.build_raw_forward_input(start_idx, end_idx);
}

void Batch::process_sample_output(const RawForwardOutput& raw_output,
                                  bool replace_fake_token) {
  // if raw_output.outputs.size() value is 0,
  // this means all sequences are in prefill stage status.
  const int64_t num_seqs = raw_output.outputs.size();
  int64_t output_idx = 0;
  for (auto* seq : sequences_) {
    if (seq->finished()) {
      output_idx++;
      continue;
    }
    if (update_sequence_state(seq, replace_fake_token)) {
      continue;
    }
    CHECK_LT(output_idx, num_seqs);

    const auto curr_idx = output_idx++;
    const RawSampleOutput raw_sam_output = raw_output.outputs[curr_idx];
    const size_t token_size = raw_sam_output.tokens.size();
    for (size_t t_idx = 0; t_idx < token_size; t_idx++) {
      Token t(raw_sam_output.tokens[t_idx].id);
      if (raw_sam_output.tokens[t_idx].logprob.has_value()) {
        t.logprob = raw_sam_output.tokens[t_idx].logprob.value();
      }
      t.top_tokens = raw_sam_output.tokens[t_idx].top_tokens;
      t.top_logprobs = raw_sam_output.tokens[t_idx].top_logprobs;
      // always append a token, maybe true or fake token
      append_token_for_sequence(seq, t, t_idx, replace_fake_token);

      if (raw_sam_output.tokens[t_idx].embeddings.size() > 0) {
        torch::Tensor embeddings =
            torch::tensor(raw_sam_output.tokens[t_idx].embeddings);
        seq->update_embeddings(embeddings);
      }
      // Speculative decoding may append an EOS token at the beginning,
      // followed by bonus tokens, causing the sequence stopping check to fail.
      if (seq->finished()) {
        break;
      }
    }
  }
  CHECK_EQ(output_idx, num_seqs);

  if (!FLAGS_enable_schedule_overlap || replace_fake_token) {
    process_beam_search();
  }
}

void Batch::process_sample_output(const SampleOutput& sample_output,
                                  bool replace_fake_token) {
  if (sample_output.embeddings.defined()) {
    const int64_t num_seqs = sample_output.embeddings.size(0);
    int64_t output_idx = 0;
    for (auto* seq : sequences_) {
      CHECK_LT(output_idx, num_seqs);
      auto cur_seq_embed =
          safe_to(sample_output.embeddings[output_idx++], torch::kFloat32);
      seq->update_embeddings(cur_seq_embed);
    }
  }

  // if sample_output.next_tokens not defined,
  // sample_output.next_tokens.size(0) value is 0,
  // this means all sequences are in prefill stage status.
  const int64_t num_seqs = sample_output.next_tokens.size(0);
  int64_t output_idx = 0;
  for (auto* seq : sequences_) {
    if (seq->finished()) {
      output_idx++;
      continue;
    }
    if (update_sequence_state(seq, replace_fake_token)) {
      continue;
    }
    CHECK_LT(output_idx, num_seqs);

    const auto curr_idx = output_idx++;
    const auto token = build_token(curr_idx,
                                   sample_output.next_tokens,
                                   sample_output.logprobs,
                                   sample_output.top_tokens,
                                   sample_output.top_logprobs);

    // always append a token, maybe true or fake token
    append_token_for_sequence(seq, token, 0, replace_fake_token);
  }
  CHECK_EQ(output_idx, num_seqs);

  if (!FLAGS_enable_schedule_overlap || replace_fake_token) {
    process_beam_search();
  }
}

bool Batch::update_sequence_state(Sequence* seq, bool replace_fake_token) {
  // In chunked prefill case, if enable_schedule_overlap, we need the
  // prefill-or-not state of last stage, otherwise, we need the state
  // of current stage.
  if (FLAGS_enable_chunked_prefill) {
    if (!replace_fake_token && seq->is_prefill_stage()) {
      seq->pre_scheduled_step_prefill_queue().push(true);
      // if not replace_fake_token, pop out here to avoid endless growth
      if (seq->pre_scheduled_step_prefill_queue().size() > 2) {
        seq->pre_scheduled_step_prefill_queue().pop();
      }
      return true;
    } else if (replace_fake_token &&
               seq->pre_scheduled_step_prefill_queue().front()) {
      seq->pre_scheduled_step_prefill_queue().pop();
      return true;
    }
  }
  return false;
}

void Batch::append_token_for_sequence(Sequence* seq,
                                      const Token& token,
                                      int token_idx,
                                      bool replace_fake_token) {
  // always append a token, maybe true or fake token
  if (!replace_fake_token) {
    seq->append_token(token);
    if (FLAGS_enable_chunked_prefill) {
      seq->pre_scheduled_step_prefill_queue().push(false);
      // if not replace_fake_token, pop out here to avoid endless growth
      if (seq->pre_scheduled_step_prefill_queue().size() > 2) {
        seq->pre_scheduled_step_prefill_queue().pop();
      }
    }
  } else {
    // truely update the real token if replace_fake_token
    seq->update_last_step_token(token, token_idx);
    if (FLAGS_enable_chunked_prefill && token_idx == 0) {
      seq->pre_scheduled_step_prefill_queue().pop();
    }
  }
}

void Batch::process_embedding_output(const torch::Tensor& output_embedding) {
  Token token(0);
  if (output_embedding.defined()) {
    int32_t slice_img_index = 0;
    for (auto* seq : sequences_) {  // TODO
      const auto& mm_data = seq->get_mm_data();

      auto pixel_values = mm_data.get_tensor_vec("pixel_values");
      constexpr const int channel = 3;
      int32_t slice_num = 0;
      for (const auto& item : pixel_values) {
        slice_num += item.size(0) / channel;
      }

      auto seq_img_embedding =
          output_embedding
              .slice(0, slice_img_index, slice_img_index + slice_num)
              .clone();
      ;
      slice_img_index += slice_num;
      seq->update_embeddings(seq_img_embedding);
      seq->append_token(token);
    }
  }
}

void Batch::process_beam_search() {
  for (auto* sequence_group : sequence_groups_) {
    sequence_group->process_beam_search();
  }
}

void Batch::process_beam_search_output(const RawForwardOutput& raw_output,
                                       bool replace_fake_token) {
  const int32_t beam_width = sequences_[0]->sampling_param()->beam_width;
  if (beam_width <= 1) {
    return;
  }

  CHECK_EQ(raw_output.src_seq_idxes.size(), sequences_.size());
  CHECK_EQ(raw_output.out_tokens.size(), sequences_.size());
  CHECK_EQ(raw_output.out_logprobs.size(), sequences_.size());

  auto update_for_sequence_group = [&](size_t sequence_group_id) {
    std::unordered_set<int32_t> seq_idx_set;
    std::vector<float> src_acc_logprob_vec;
    std::vector<std::vector<int32_t>> src_token_ids;
    std::vector<std::vector<std::optional<float>>> src_logprobs;
    src_acc_logprob_vec.resize(beam_width);
    src_token_ids.resize(beam_width);
    src_logprobs.resize(beam_width);

    for (size_t i = 0; i < beam_width; i++) {
      size_t task_id = sequence_group_id * beam_width + i;
      int32_t src_seq_idx = raw_output.src_seq_idxes[task_id];
      CHECK_LE(src_seq_idx, sequences_.size());
      auto src_seq = sequences_[src_seq_idx];
      src_acc_logprob_vec[i] =
          src_seq->get_average_logprob() * src_seq->num_generated_tokens();
      src_token_ids[i] = std::vector<int32_t>(src_seq->tokens());
      src_logprobs[i] = src_seq->logprob_state()->get_logprobs();
    }

    for (size_t i = 0; i < beam_width; i++) {
      size_t task_id = sequence_group_id * beam_width + i;
      int32_t src_seq_idx = raw_output.src_seq_idxes[task_id];
      CHECK_LE(src_seq_idx, sequences_.size());
      auto& base_seq = sequences_[task_id];
      auto& src_seq = sequences_[src_seq_idx];

      for (size_t token_idx = base_seq->num_prompt_tokens();
           token_idx < base_seq->num_tokens();
           token_idx++) {
        Token new_token(src_token_ids[i][token_idx]);
        new_token.logprob = src_logprobs[i][token_idx];
        base_seq->update_token(token_idx, new_token);
      }

      Token new_token(raw_output.out_tokens[task_id]);
      new_token.logprob =
          raw_output.out_logprobs[task_id] - src_acc_logprob_vec[i];
      append_token_for_sequence(base_seq, new_token, 0, replace_fake_token);

      base_seq->logprob_state()->set_acc_logprob(
          raw_output.out_logprobs[task_id]);
      base_seq->logprob_state()->set_last_acc_token_idx(base_seq->num_tokens());

      bool need_swap = false;
      if (seq_idx_set.find(src_seq_idx) != seq_idx_set.end()) {
        need_swap = true;
      } else {
        seq_idx_set.insert(src_seq_idx);
      }

      auto src_blocks = src_seq->kv_state().kv_blocks();
      base_seq->kv_state().set_src_blocks(src_blocks, need_swap);
    }
  };

  for (size_t sequence_group_id = 0;
       sequence_group_id < sequence_groups_.size();
       sequence_group_id++) {
    update_for_sequence_group(sequence_group_id);
  }
}

void Batch::finish() {
  // Finish all sequence groups
  for (auto* sequence_group : sequence_groups_) {
    sequence_group->finish();
  }
}
}  // namespace xllm
