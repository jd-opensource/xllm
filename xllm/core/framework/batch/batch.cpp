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
#include "runtime/params_utils.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {

namespace {
void split_copy_out_blocks(RawModelInput& raw_model_input,
                           std::unordered_set<int32_t>& write_block_ids) {
  std::vector<CacheContent> async_copy_out_blocks;
  std::vector<CacheContent> sync_copy_out_blocks;
  for (CacheContent content : raw_model_input.copy_out_blocks) {
    if (write_block_ids.find(content.device_block_id) !=
        write_block_ids.end()) {
      sync_copy_out_blocks.push_back(content);
    } else {
      async_copy_out_blocks.push_back(content);
    }
  }
  raw_model_input.copy_out_blocks = std::move(sync_copy_out_blocks);
  raw_model_input.async_copy_out_blocks = std::move(async_copy_out_blocks);
}
}  // namespace

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
  BatchInputBuilder builder(sequences_,
                            allowed_max_tokens_,
                            input_embeddings_vec_,
                            mm_data_vec_,
                            &args);
  return builder.build_forward_input(num_decoding_tokens,
                                     min_decoding_batch_size);
}

RawForwardInput Batch::prepare_forward_input() {
  BatchInputBuilder builder(sequences_,
                            allowed_max_tokens_,
                            input_embeddings_vec_,
                            mm_data_vec_,
                            nullptr);
  return builder.build_raw_forward_input();
}

void Batch::process_sample_output(const RawForwardOutput& raw_output,
                                  bool enable_schedule_overlap) {
  // if raw_output.outputs.size() value is 0,
  // this means all sequences are in prefill stage status.
  const int64_t num_seqs = raw_output.outputs.size();
  int64_t output_idx = 0;
  for (auto* seq : sequences_) {
    if (seq->finished()) {
      output_idx++;
      continue;
    }
    if (update_sequence_state(seq, enable_schedule_overlap)) {
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
      append_token_for_sequence(seq, t, t_idx, enable_schedule_overlap);

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
}

void Batch::process_sample_output(const SampleOutput& sample_output,
                                  bool enable_schedule_overlap) {
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
    if (update_sequence_state(seq, enable_schedule_overlap)) {
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
    append_token_for_sequence(seq, token, 0, enable_schedule_overlap);
  }
  CHECK_EQ(output_idx, num_seqs);
}

bool Batch::update_sequence_state(Sequence* seq, bool enable_schedule_overlap) {
  // In chunked prefill case, if enable_schedule_overlap, we need the
  // prefill-or-not state of last stage, otherwise, we need the state
  // of current stage.
  if (FLAGS_enable_chunked_prefill) {
    if (!enable_schedule_overlap && seq->is_prefill_stage()) {
      seq->pre_scheduled_step_prefill_queue().push(true);
      // if not enable_schedule_overlap, pop out here to avoid endless growth
      if (seq->pre_scheduled_step_prefill_queue().size() > 2) {
        seq->pre_scheduled_step_prefill_queue().pop();
      }
      return true;
    } else if (enable_schedule_overlap &&
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
                                      bool enable_schedule_overlap) {
  // always append a token, maybe true or fake token
  if (!enable_schedule_overlap) {
    seq->append_token(token);
    if (FLAGS_enable_chunked_prefill) {
      seq->pre_scheduled_step_prefill_queue().push(false);
      // if not enable_schedule_overlap, pop out here to avoid endless growth
      if (seq->pre_scheduled_step_prefill_queue().size() > 2) {
        seq->pre_scheduled_step_prefill_queue().pop();
      }
    }
  } else {
    // truely update the real token if enable_schedule_overlap
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
}  // namespace xllm
