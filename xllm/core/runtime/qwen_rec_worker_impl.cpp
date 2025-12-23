/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================*/

#include "qwen_rec_worker_impl.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "core/common/global_flags.h"
#include "framework/model/model_input_params.h"
#include "util/timer.h"
#if defined(USE_NPU)
#include "kernels/npu/xllm_ops/beam_search_group.h"
#include "kernels/npu/xllm_ops/cache_select.h"
#endif

namespace xllm {

QwenRecWorkerImpl::QwenRecWorkerImpl(const ParallelArgs& parallel_args,
                                     const torch::Device& device,
                                     const runtime::Options& options)
    : LLMWorkerImpl(parallel_args, device, options) {}

std::optional<ForwardOutput> QwenRecWorkerImpl::step(
    const ForwardInput& input) {
  return step_multi_round(input);
}

std::optional<ForwardOutput> QwenRecWorkerImpl::step_multi_round(
    ForwardInput input) {
  device_.set_device();
  Timer timer;

  int32_t total_rounds = input.total_round;
  ForwardOutput output;

  std::vector<torch::Tensor> unshared_k_cache;
  std::vector<torch::Tensor> unshared_v_cache;
  auto args = context_.get_model_args();
  int32_t layer_num = static_cast<int32_t>(args.n_layers());
  for (auto i = 0; i < layer_num; ++i) {
    unshared_k_cache.push_back(kv_caches_[i].get_k_cache());
    unshared_v_cache.push_back(kv_caches_[i].get_v_cache());
  }
  int32_t batch = input.input_params.num_sequences;

  // NOTE:
  // - beam_width for multi-round decode is carried in ForwardInput::beam_width,
  //   which is filled from proto (see proto_to_forward_input).
  // - ModelInputParams::beam_width is not populated by the multi-step builders
  //   on host side.
  int32_t beam_width_init = input.beam_width;
  input.input_params.beam_width = beam_width_init;

  auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device_);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  torch::Tensor sequence_group =
      torch::zeros({batch, beam_width_init, total_rounds}, int_options);
  // preallocate outputs and cached inputs
  int64_t num_seq = batch * beam_width_init;
  torch::Tensor acc_logprob = torch::empty({num_seq, 1}, fp32_options);
  torch::Tensor out_log_probs = torch::empty({num_seq, 1}, fp32_options);
  torch::Tensor out_token_ids = torch::empty({num_seq, 1}, int_options);
  torch::Tensor out_token_index = torch::empty({num_seq, 1}, int_options);
  torch::Tensor out_beam_count_prefix_sums =
      torch::empty({num_seq, 1}, int_options);
  auto out_seqgroup = sequence_group.clone();

  for (int32_t round = 0; round < total_rounds; ++round) {
    const auto& sampling_params =
        round > 0 ? input.decoder_sampling_params : input.sampling_params;
    input.input_params.is_prefill = (round == 0);
    if (!input.input_params.current_round_tensor_list.empty() && round >= 0 &&
        round < static_cast<int32_t>(
                    input.input_params.current_round_tensor_list.size())) {
      input.input_params.current_round_tensor =
          input.input_params.current_round_tensor_list[round];
      input.input_params.current_round = round;
    }

    auto hidden_states = model_executor_->forward(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    if (!hidden_states.defined()) {
      return std::nullopt;
    }

    torch::Tensor logits;

    if (sampling_params.selected_token_idxes.defined()) {
      logits =
          model_->logits(hidden_states, sampling_params.selected_token_idxes);
    }
    if (sampling_params.selected_token_idxes.defined()) {
      auto sample_output = sampler_->forward(logits, sampling_params);
      torch::Tensor top_tokens;
      torch::Tensor top_logprobs;
      int32_t beam_width = beam_width_init;

      if (round == 0) {
        top_tokens =
            sample_output.top_tokens.to(torch::kInt32).reshape({-1, 1});
        top_logprobs = sample_output.top_logprobs.reshape({-1, 1});
      } else {
        top_tokens = sample_output.top_tokens.to(torch::kInt32)
                         .reshape({-1, beam_width});
        top_logprobs = sample_output.top_logprobs.reshape({-1, beam_width});
      }
#if defined(USE_NPU)
      xllm_ops::beam_search(acc_logprob,
                            top_tokens,
                            top_logprobs,
                            sequence_group,
                            round,
                            out_token_ids,
                            out_token_index,
                            out_log_probs,
                            out_beam_count_prefix_sums,
                            out_seqgroup);
#endif
      sequence_group.copy_(out_seqgroup);
      acc_logprob.copy_(out_log_probs);
      // keep group offset contiguous across rounds (already in out_* tensors)
      // update next round tokens.
      if (round == 0) {
        input.token_ids =
            sample_output.top_tokens.to(torch::kInt32).reshape({-1});
      } else {
        input.token_ids = out_token_ids.clone().reshape({-1});
      }

      // update next round positions.
      if (!input.input_params.decode_positions_tensor_list.empty() &&
          round >= 0 &&
          round < static_cast<int32_t>(
                      input.input_params.decode_positions_tensor_list.size())) {
        input.positions =
            input.input_params.decode_positions_tensor_list[round];
      }

      // update output at the last round.
      if (round == total_rounds - 1) {
        output.logits = logits;
        output.sample_output = sample_output;
        output.do_sample = sampling_params.do_sample;
        output.logprobs = sampling_params.logprobs;
        output.max_top_logprobs = sampling_params.max_top_logprobs;
        output.beam_search_output.src_seq_idxes = out_token_index.reshape({-1});
        output.beam_search_output.out_tokens = out_token_ids.reshape({-1});
        output.beam_search_output.out_logprobs = out_log_probs.reshape({-1});
        output.beam_search_output.group_offset =
            out_beam_count_prefix_sums.reshape({-1});
        output.beam_sequence_group = sequence_group;
      }

#if defined(USE_NPU)
      if (beam_width > 1 && round > 0) {
        xllm_ops::cache_select(out_token_index,
                               unshared_k_cache,
                               unshared_v_cache,
                               input.input_params.block_tables,
                               out_beam_count_prefix_sums,
                               round,
                               beam_width,
                               layer_num);
      }
#endif
    }
  }

  auto ret = device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());
  return output;
}

}  // namespace xllm
