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

#include "rec_worker_impl.h"

#include <glog/logging.h>

#include <chrono>
#include <exception>
#include <utility>

#include "butil/file_util.h"
#include "butil/files/dir_reader_linux.h"
#include "butil/files/file_path.h"
#include "butil/strings/string_util.h"
#include "common/metrics.h"
#include "models/model_registry.h"
#include "util/env_var.h"
#include "util/utils.h"

namespace xllm {

RecWorkerImpl::RecWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  // Initialize filter mask stream for H2D operations
  filter_mask_stream_ = device_.get_stream_from_pool();

  // Initialize thread pool for async operations using environment variable
  int thread_num = util::get_int_env(util::EXTRA_THREAD_NUM, 16);
  thread_pool_ = std::make_shared<ThreadPool>(thread_num);
}

bool RecWorkerImpl::init_model(const std::string& model_weights_path) {
  auto model_loader = ModelLoader::create(model_weights_path);

  auto args = model_loader->model_args();
  auto quant_args = model_loader->quant_args();
  torch::ScalarType dtype = util::parse_dtype(args.dtype(), device_);

  if (options_.enable_speculative_decode() && FLAGS_enable_atb_spec_kernel) {
    args.num_speculative_tokens(options_.num_speculative_tokens());
  }

  // create model context
  dtype_ = dtype;
  auto tensor_options = torch::dtype(dtype_).device(device_);
  context_ = ModelContext(parallel_args_, args, quant_args, tensor_options);

  // init model, create model executor
  bool status = this->init_model(context_);
  if (!status) {
    return false;
  }

  this->load_model(std::move(model_loader));

  status_ = Status::LOADED;
  // TODO: replace path with flags after filter merge
  butil::FilePath filter_bin_path =
      butil::FilePath(model_weights_path).Append("replace me when merge");
  valid_path_filter_ = std::make_unique<ValidPathFilter>(
      filter_bin_path.value(), args.vocab_size(), dtype_, device_);

  return true;
}

bool RecWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";
  device_.set_device();

  // Try to create a causal LM model (Rec models are typically based on
  // CausalLM)
  model_ = create_llm_model(context);

  // Check if model creation was successful
  CHECK(model_ != nullptr) << "Failed to create Rec model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }
  return true;
}

std::optional<ForwardOutput> RecWorkerImpl::step(
    const BatchedForwardInputs& inputs) {
  device_.set_device();

  // Timer for performance monitoring
  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<torch::Tensor> flatten_tokens_micro_batches;
  std::vector<torch::Tensor> flatten_positions_micro_batches;
  std::vector<ModelInputParams> input_params_micro_batches;
  auto& concated_sampling_params = inputs.concated_sampling_params;

  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    flatten_tokens_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].token_ids));
    flatten_positions_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].positions));
    input_params_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].input_params));
  }
  auto sampling_params = inputs.micro_inputs[0].sampling_params;
  // Start async filter mask preparation early for overlap (if beam search is
  // enabled)
  std::future<torch::Tensor> filter_mask_future;

  if (!input_params_micro_batches.empty() &&
      input_params_micro_batches[0].is_rec_model() &&
      input_params_micro_batches[0].rec_params.has_value()) {
    auto& rec_params = input_params_micro_batches[0].rec_params.value();
    if (!rec_params.generated_tokens.empty()) {
      filter_mask_future =
          prepare_filter_mask_async(rec_params.generated_tokens);
    }
  }

  // Check if we have encoder inputs (rec model with encoder/decoder)
  torch::Tensor hidden_states;
  bool has_encoder_inputs = false;

  // Check if this is a rec model with encoder inputs
  if (!input_params_micro_batches.empty() &&
      input_params_micro_batches[0].is_rec_model() &&
      input_params_micro_batches[0].rec_params.has_value()) {
    auto& rec_params = input_params_micro_batches[0].rec_params.value();

    // Check for encoder inputs
    if ((rec_params.encoder_token_ids.defined() &&
         rec_params.encoder_positions.defined()) ||
        rec_params.encoder_sparse_embedding.defined()) {
      has_encoder_inputs = true;

      // Set hybrid mode if sparse embedding is defined
      if (rec_params.encoder_sparse_embedding.defined()) {
        input_params_micro_batches[0].rec_params->is_hybrid_mode = true;
      }
    }
  }

  // Two-stage forward: encoder then decoder
  auto& rec_params = input_params_micro_batches[0].rec_params.value();

  if (rec_params.rec_stage == RecModelInputParams::RecStage::PREFILL) {
    // Check if this is the first prefill or subsequent prefill
    if (!rec_params.is_first_prefill) {
      // Subsequent prefill: only run decoder
      input_params_micro_batches[0].rec_params->is_encoder_forward = false;
      hidden_states = model_executor_->forward(flatten_tokens_micro_batches,
                                               flatten_positions_micro_batches,
                                               kv_caches_,
                                               input_params_micro_batches);
    } else if (has_encoder_inputs) {
      // First prefill: run encoder first, then decoder

      // 1. Run encoder forward
      auto encoder_input_params = input_params_micro_batches;
      encoder_input_params[0].rec_params->is_encoder_forward = true;

      std::vector<torch::Tensor> encoder_tokens;
      std::vector<torch::Tensor> encoder_positions;

      if (rec_params.is_hybrid_mode &&
          rec_params.encoder_sparse_embedding.defined()) {
        encoder_tokens.push_back(rec_params.encoder_sparse_embedding);
      } else {
        encoder_tokens.push_back(rec_params.encoder_token_ids);
      }
      encoder_positions.push_back(rec_params.encoder_positions);

      // Run encoder
      hidden_states = model_executor_->forward(
          encoder_tokens, encoder_positions, kv_caches_, encoder_input_params);

      // 2. Run decoder forward
      encoder_input_params[0].rec_params->is_encoder_forward = false;
      hidden_states = model_executor_->forward(flatten_tokens_micro_batches,
                                               flatten_positions_micro_batches,
                                               kv_caches_,
                                               encoder_input_params);

    } else {
      // Non-rec model or rec model without encoder: use standard forward
      LOG(ERROR) << "RecWorkerImpl not supports decoder-only model now.";
    }
  } else {
    // Decode stage: only run decoder, not used now.
    hidden_states = model_executor_->forward(flatten_tokens_micro_batches,
                                             flatten_positions_micro_batches,
                                             kv_caches_,
                                             input_params_micro_batches);
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits =
        model_->logits(hidden_states, sampling_params.selected_token_idxes);
  }

  ForwardOutput output;

  if (!driver_) {
    return std::nullopt;
  }

  // Get filter mask result from async preparation if available
  torch::Tensor filter_mask;
  if (filter_mask_future.valid()) {
    // Get the result from async preparation (this will block if not ready)
    filter_mask = filter_mask_future.get();
  }

  // Driver prepare model output

  if (sampling_params.selected_token_idxes.defined()) {
    // auto sample_logits =
    //     logits.index_select(/*dim=*/0,
    //     concated_sampling_params.sample_idxes);

    // Apply filter mask if available
    // TODO: fix filter
    // if (filter_mask.defined()) {
    //   // Ensure filter_mask has the same batch size as sample_logits
    //   if (filter_mask.size(0) == sample_logits.size(0)) {
    //     sample_logits = sample_logits + filter_mask;
    //   } else {
    //     // If dimensions don't match, select the appropriate rows from
    //     // filter_mask
    //     auto selected_filter_mask = filter_mask.index_select(
    //         /*dim=*/0, concated_sampling_params.sample_idxes);
    //     sample_logits = sample_logits + selected_filter_mask;
    //   }
    // }

    auto sample_output = sampler_->forward(logits, sampling_params);
    output.logits = logits;

    // Set sample output to output
    output.sample_output = sample_output;

    // Carry over the sampling params
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }

  // Transfer sample output tensors to CPU for batch.cpp access
  if (output.sample_output.next_tokens.defined()) {
    output.sample_output.next_tokens =
        safe_to(output.sample_output.next_tokens, torch::kCPU, true);
  }
  if (output.sample_output.logprobs.defined()) {
    output.sample_output.logprobs =
        safe_to(output.sample_output.logprobs, torch::kCPU, true);
  }
  if (output.sample_output.top_tokens.defined()) {
    output.sample_output.top_tokens =
        safe_to(output.sample_output.top_tokens, torch::kCPU, true);
  }
  if (output.sample_output.top_logprobs.defined()) {
    output.sample_output.top_logprobs =
        safe_to(output.sample_output.top_logprobs, torch::kCPU, true);
  }

  // Synchronize at the end like in llm_worker_impl
  auto ret = device_.synchronize_default_stream();

  // Record execution latency
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  COUNTER_ADD(execution_latency_seconds_model, duration.count() / 1000000.0);

  return output;
}

ForwardInput RecWorkerImpl::prepare_inputs(Batch& batch) {
  // Use the rec-specific input preparation method
  return batch.prepare_rec_forward_input(options_.num_decoding_tokens(),
                                         0,  // min_decoding_batch_size
                                         context_.get_model_args());
}

std::future<torch::Tensor> RecWorkerImpl::prepare_filter_mask_async(
    const std::vector<std::vector<int32_t>>& generated_tokens) {
  // Create promise/future pair for async result
  auto promise = std::make_shared<std::promise<torch::Tensor>>();
  auto future = promise->get_future();

  // Submit async task to thread pool
  thread_pool_->schedule([this, generated_tokens, promise]() -> void {
    try {
      // Set stream guard for H2D operations
      c10::StreamGuard streamGuard = filter_mask_stream_->set_stream_guard();

      torch::Tensor cpu_mask;

      // Use ValidPathFilter if available, otherwise create placeholder mask
      if (valid_path_filter_ && !generated_tokens.empty()) {
        // Use ValidPathFilter to generate the actual filter mask
        cpu_mask = valid_path_filter_->forward(generated_tokens);

        // If ValidPathFilter returns empty tensor, create placeholder
        if (!cpu_mask.defined()) {
          int batch_size = generated_tokens.size();
          int vocab_size = 32000;  // Default vocab size
          cpu_mask = torch::zeros({batch_size, vocab_size}, torch::kFloat32);
        }
      } else if (!generated_tokens.empty()) {
        // Fallback: create placeholder mask when ValidPathFilter is not
        // available
        int batch_size = generated_tokens.size();
        int vocab_size = 32000;  // Default vocab size
        cpu_mask = torch::zeros({batch_size, vocab_size}, torch::kFloat32);

        // Apply some basic filtering logic (placeholder)
        for (int i = 0; i < batch_size; ++i) {
          // Set some tokens to -inf to filter them out
          cpu_mask[i]
              .slice(0, 0, 1000)
              .fill_(-std::numeric_limits<float>::infinity());
        }
      } else {
        // Return empty tensor if no generated tokens
        promise->set_value(torch::Tensor());
        return;
      }

      // Copy to device using the dedicated H2D stream
      torch::Tensor device_mask = cpu_mask.to(device_, /*non_blocking=*/true);

      // Synchronize the H2D stream to ensure copy is complete
      filter_mask_stream_->synchronize();

      // Set the result in the promise
      promise->set_value(device_mask);
    } catch (const std::exception& e) {
      // Set exception in promise if something goes wrong
      promise->set_exception(std::current_exception());
    }
  });

  return future;
}

}  // namespace xllm