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

#include "worker_impl.h"

#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>
#if defined(USE_NPU)
#include "kernels/npu/xllm_ops/replace_token.h"
#elif defined(USE_MLU)
#include <torch_mlu/csrc/framework/core/caching_allocator.h>
#endif

#include <memory>
#include <optional>
#include <utility>

#include "common/device_monitor.h"
#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_loader.h"
#include "framework/sampling/sampler.h"
#include "framework/state_dict/state_dict.h"
#include "framework/xtensor/multi_layer_xtensor_transfer.h"
#include "util/net.h"
#include "util/tensor_helper.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

constexpr uint64_t MBUF_SIZE = 128 * 1024 * 1024;

WorkerImpl::WorkerImpl(const ParallelArgs& parallel_args,
                       const torch::Device& device,
                       const runtime::Options& options)
    : options_(options), device_(device), parallel_args_(parallel_args) {
  if (options_.enable_speculative_decode() &&
      options_.num_decoding_tokens() == 1) {
    is_spec_draft_ = true;
  }

  // first worker is the driver
  driver_ = parallel_args.rank() == 0;
  int32_t tp_size = parallel_args.world_size() / parallel_args.dp_size();
  dp_driver_ =
      parallel_args.dp_size() > 1 && parallel_args.rank() % tp_size == 0;

  device_.set_device();
  device_.init_device_context();
  threadpool_.schedule([this]() mutable { device_.set_device(); });
  general_threadpool_.schedule([this]() mutable { device_.set_device(); });

  prepare_stream_ = device_.get_stream_from_pool();
  copy_out_stream_ = device_.get_stream_from_pool();
  sampler_ = std::make_unique<Sampler>();
}

WorkerImpl::~WorkerImpl() = default;

bool WorkerImpl::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(kv_caches_.empty()) << "KV caches are already initialized.";

  // create a KVCache for each layer
  const int64_t num_layers = context_.get_model_args().n_layers();
  kv_caches_.reserve(num_layers);
  for (int64_t i = 0; i < num_layers; ++i) {
    torch::Tensor key_cache, value_cache;
#if defined(USE_NPU)
    key_cache = at_npu::native::npu_format_cast(
        torch::empty(kv_cache_shape[0], torch::dtype(dtype_).device(device_)),
        2);
    value_cache = at_npu::native::npu_format_cast(
        torch::empty(kv_cache_shape[1], torch::dtype(dtype_).device(device_)),
        2);
#elif defined(USE_MLU)
    key_cache =
        torch::empty(kv_cache_shape[0], torch::dtype(dtype_).device(device_));
    value_cache =
        torch::empty(kv_cache_shape[1], torch::dtype(dtype_).device(device_));
#endif
    kv_caches_.emplace_back(key_cache, value_cache);
  }

  allocate_host_kv_cache(kv_cache_shape);
  status_ = Status::READY;
  return true;
}

bool WorkerImpl::allocate_host_kv_cache(
    const std::vector<std::vector<int64_t>>& device_kv_cache_shape) {
  if (options_.host_blocks_factor() <= 0.00001) {
    return true;
  }

  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(host_kv_caches_.empty()) << "KV caches are already initialized.";

  std::vector<std::vector<int64_t>> host_kv_cache_shape = device_kv_cache_shape;
  host_kv_cache_shape[0][0] =
      device_kv_cache_shape[0][0] * options_.host_blocks_factor();
  host_kv_cache_shape[1][0] =
      device_kv_cache_shape[1][0] * options_.host_blocks_factor();

  // create a KVCache for each layer
  const int64_t num_layers = context_.get_model_args().n_layers();
  kv_caches_.reserve(num_layers);
  for (int64_t i = 0; i < num_layers; ++i) {
    torch::Tensor key_cache, value_cache;
    key_cache = torch::empty(host_kv_cache_shape[0],
                             torch::dtype(dtype_).device(torch::kCPU));
    value_cache = torch::empty(host_kv_cache_shape[1],
                               torch::dtype(dtype_).device(torch::kCPU));
    host_kv_caches_.emplace_back(key_cache, value_cache);
  }

  if (options_.enable_kvcache_store()) {
    StoreConfig config;
    config.protocol = options_.store_protocol();
    config.metadata_connstring = options_.store_metadata_connstring();
    config.master_server_entry = options_.store_master_server_entry();
    config.tp_rank = options_.node_rank() % options_.dp_size();

    kv_cache_store_ = std::make_shared<KVCacheStore>(config, &host_kv_caches_);
  }

  status_ = Status::READY;
  return true;
}

bool WorkerImpl::allocate_continuous_kv_cache(
    const std::vector<XTensor::Options>& options) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(kv_caches_.empty()) << "KV caches are already initialized.";

  // create a KVCache for each layer
  const int64_t num_layers = context_.get_model_args().n_layers();
  kv_caches_.reserve(num_layers);

  std::shared_ptr<XTensor> key_xtensor;
  std::shared_ptr<XTensor> value_xtensor;

  std::vector<std::shared_ptr<XTensor>> key_xtensors(num_layers);
  std::vector<std::shared_ptr<XTensor>> value_xtensors(num_layers);

  for (int64_t i = 0; i < num_layers; ++i) {
    key_xtensor = std::make_shared<XTensor>(options[0], dtype_);
    key_xtensors[i] = key_xtensor;

    value_xtensor = std::make_shared<XTensor>(options[1], dtype_);
    value_xtensors[i] = value_xtensor;

    kv_caches_.emplace_back(key_xtensor, value_xtensor);
  }

  MultiLayerXTensorTransfer::get_instance().set_multi_layer_xtensor(
      key_xtensors, value_xtensors, device_);

  status_ = Status::READY;
  return true;
}

bool WorkerImpl::allocate_kv_cache_with_transfer(
    uint64_t kv_cache_size,
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
#if defined(USE_NPU)
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(kv_caches_.empty()) << "KV caches are already initialized.";

  int32_t device_id = device_.index();
  if (FLAGS_kv_cache_transfer_type == "LlmDataDist") {
    kv_cache_transfer_ =
        std::make_shared<LlmDataDistTransfer>(options_.device_ip().value(),
                                              options_.transfer_listen_port(),
                                              options_.instance_role());

    // create a KVCache for each layer
    const int64_t num_layers = context_.get_model_args().n_layers();
    kv_caches_.reserve(num_layers);

    int32_t device_id = device_.index();
    kv_cache_transfer_->initialize(device_id);
    kv_cache_transfer_->allocate_kv_cache(
        kv_caches_, num_layers, kv_cache_shape, dtype_);
  } else {
    kv_cache_transfer_ = std::make_unique<HcclKVCacheTransfer>(
        device_id, options_.transfer_listen_port());

    allocate_kv_cache(kv_cache_shape);
    kv_cache_transfer_->register_kv_cache(kv_caches_, kv_cache_shape, dtype_);
  }
#endif

  allocate_host_kv_cache(kv_cache_shape);
  status_ = Status::READY;
  return true;
}

#if defined(USE_NPU)
bool WorkerImpl::allocate_kv_cache_with_transfer(
    std::shared_ptr<KVCacheTransfer> kv_cache_transfer,
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(kv_caches_.empty()) << "KV caches are already initialized.";

  kv_cache_transfer_ = kv_cache_transfer;

  // create a KVCache for each layer
  const int64_t num_layers = context_.get_model_args().n_layers();
  kv_caches_.reserve(num_layers);
  if (is_spec_draft_) {
    kv_cache_transfer_->allocate_kv_cache_spec(
        kv_caches_, num_layers, kv_cache_shape, dtype_);
  } else {
    kv_cache_transfer_->allocate_kv_cache(
        kv_caches_, num_layers, kv_cache_shape, dtype_);
  }

  allocate_host_kv_cache(kv_cache_shape);
  status_ = Status::READY;
  return true;
}
#endif

void WorkerImpl::get_device_info(std::string& device_ip, uint16_t& port) {
  // device_ip = options_.device_ip().value();
  device_ip = net::get_local_ip_addr();
  port = options_.transfer_listen_port();
}

void WorkerImpl::get_cache_info(uint64_t& cluster_id,
                                std::string& addr,
                                int64_t& k_cache_id,
                                int64_t& v_cache_id) {
#if defined(USE_NPU)
  kv_cache_transfer_->get_cache_info(cluster_id, addr, k_cache_id, v_cache_id);
#endif
}

bool WorkerImpl::link_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<std::string>& device_ips,
                              const std::vector<uint16_t>& ports) {
#if defined(USE_NPU)
  for (int32_t i = 0; i < cluster_ids.size(); ++i) {
    if (!kv_cache_transfer_->link_cluster(
            cluster_ids[i], addrs[i], device_ips[i], ports[i])) {
      return false;
    }
  }
#endif
  return true;
}

bool WorkerImpl::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                const std::vector<std::string>& addrs,
                                const std::vector<std::string>& device_ips,
                                const std::vector<uint16_t>& ports) {
#if defined(USE_NPU)
  for (int32_t i = 0; i < cluster_ids.size(); ++i) {
    if (!kv_cache_transfer_->unlink_cluster(
            cluster_ids[i], addrs[i], device_ips[i], ports[i])) {
      return false;
    }
  }
#endif
  return true;
}

std::tuple<int64_t, int64_t> WorkerImpl::estimate_kv_cache_capacity() {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  size_t torch_cache = 0;
  size_t torch_largest_block = 0;
  int32_t device_id = device_.index();
#if defined(USE_NPU)
  c10_npu::NPUCachingAllocator::emptyCache();
  c10_npu::NPUCachingAllocator::FreeDeviceCachedMemory(device_id);
  // aclrtSynchronizeDevice();
  // get torch's cache memory size since torch_npu's emptyCache is useless
  c10_npu::NPUCachingAllocator::cacheInfo(
      device_id, &torch_cache, &torch_largest_block);
#elif defined(USE_MLU)
  torch_mlu::MLUCachingAllocator::emptyCache();
#endif
  const auto available_memory = device_.free_memory();
  const auto total_memory = device_.total_memory();
  DeviceMonitor::get_instance().set_total_memory(device_id, total_memory);
  DeviceMonitor::get_instance().set_weight_memory(
      device_id, total_memory - available_memory - torch_cache);
  return {available_memory + torch_cache, total_memory};
}

void WorkerImpl::process_group_test() {
  device_.set_device();

  // create random tensors
  const auto options = torch::dtype(torch::kHalf).device(device_);
  torch::Tensor tensor = torch::randn({10, 10}, options);
  // call allreduce
  parallel_state::reduce(tensor, parallel_args_.process_group_);
  // call allgather
  parallel_state::gather(tensor, parallel_args_.process_group_);
}

ForwardInput WorkerImpl::prepare_inputs(Batch& batch) {
  return model_executor_->prepare_inputs(batch);
}

folly::SemiFuture<std::tuple<int64_t, int64_t>>
WorkerImpl::estimate_kv_cache_capacity_async() {
  folly::Promise<std::tuple<int64_t, int64_t>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    const auto output = this->estimate_kv_cache_capacity();
    promise.setValue(output);
  });
  return future;
}

void WorkerImpl::update_last_step_output(
    const std::optional<ForwardOutput>& output) {
  if (output.value().sample_output.next_tokens.defined()) {
    last_step_output_ = std::move(output.value());
    last_step_output_valid_ = true;
  } else {
    if (FLAGS_enable_eplb) {
      last_step_output_ = std::move(output.value());
    }
    last_step_output_valid_ = false;
  }
}

ForwardInput WorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
#if defined(USE_A3) || defined(USE_MLU)
  auto& flatten_tokens = inputs.token_ids;
  auto neg_mask = (flatten_tokens < 0);
  auto clamped_neg_indices = torch::clamp(-flatten_tokens, 0);
  auto replacement = last_step_output_.sample_output.next_tokens.index(
      {clamped_neg_indices - 1});
  inputs.token_ids = torch::where(neg_mask, replacement, flatten_tokens);
#else
  xllm_ops::replace_token(inputs.token_ids,
                          last_step_output_.sample_output.next_tokens);
#endif
  return inputs;
}

void WorkerImpl::prepare_work_before_execute(
    const BatchedForwardInputs& inputs,
    BatchedForwardInputs& processed_inputs) {
  c10::StreamGuard streamGuard = prepare_stream_->set_stream_guard();

  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    ForwardInput fwd_inputs_on_device;
    fwd_inputs_on_device = inputs.micro_inputs[i].to(device_, dtype_);
    auto& input_params = fwd_inputs_on_device.input_params;
#if defined(USE_NPU)
    if (input_params.copy_out_blocks.size() > 0 ||
        input_params.copy_in_blocks.size() > 0) {
      const int64_t num_layers = context_.get_model_args().n_layers();
      for (int layer_id = 0; layer_id < num_layers; layer_id++) {
        auto key_cache = kv_caches_[layer_id].get_k_cache();
        auto host_k_cache = host_kv_caches_[layer_id].get_k_cache();
        auto value_cache = kv_caches_[layer_id].get_v_cache();
        auto host_v_cache = host_kv_caches_[layer_id].get_v_cache();

        for (auto block_info : input_params.copy_out_blocks) {
          host_k_cache[block_info.host_block_id].copy_(
              key_cache[block_info.device_block_id]);
          host_v_cache[block_info.host_block_id].copy_(
              value_cache[block_info.device_block_id]);
        }
        for (auto block_info : input_params.copy_in_blocks) {
          key_cache[block_info.device_block_id].copy_(
              host_k_cache[block_info.host_block_id]);
          value_cache[block_info.device_block_id].copy_(
              host_v_cache[block_info.host_block_id]);
        }
      }

      offload_kv_blocks_to_store_async(
          inputs.micro_inputs[i].input_params.copy_out_blocks);

      if (input_params.swap_blocks.size() > 0 &&
          !FLAGS_enable_block_copy_kernel) {
        auto& swap_blocks = input_params.swap_blocks;

        // collect src and dst indices
        std::vector<int64_t> src_indices, dst_indices;
        src_indices.reserve(swap_blocks.size());
        dst_indices.reserve(swap_blocks.size());

        for (const auto& block : swap_blocks) {
          src_indices.push_back(block.device_block_id);
          dst_indices.push_back(block.host_block_id);
        }

        // batch select keys and values
        auto src_tensor = torch::tensor(
            src_indices, torch::dtype(torch::kLong).device(device_));
        auto dst_tensor = torch::tensor(
            dst_indices, torch::dtype(torch::kLong).device(device_));
        const int64_t num_layers = context_.get_model_args().n_layers();
        for (int layer_id = 0; layer_id < num_layers; layer_id++) {
          kv_caches_[layer_id].swap_blocks(src_tensor, dst_tensor);
        }
      }
    }
    if (!context_.get_parallel_args().mapping_data().empty()) {
      torch::Tensor token_size_per_dp_group =
          torch::tensor(fwd_inputs_on_device.input_params.dp_global_token_nums,
                        torch::TensorOptions()
                            .device(torch::kCPU)
                            .dtype(torch::kInt32)
                            .pinned_memory(true));
      bool is_prefill =
          fwd_inputs_on_device.input_params.batch_forward_type.is_prefill();
      DpEpPadding dp_ep_padding(token_size_per_dp_group,
                                context_.get_model_args().num_experts_per_tok(),
                                context_.get_parallel_args().mapping_data(),
                                device_,
                                dtype_,
                                is_prefill);
      fwd_inputs_on_device.input_params.dp_ep_padding_data =
          dp_ep_padding.build();
      if (FLAGS_enable_eplb) {
        // expert_load_data_.fill_(0);
        fwd_inputs_on_device.input_params.expert_load_data = expert_load_data_;
      }
    }
#endif
    processed_inputs.micro_inputs.push_back(std::move(fwd_inputs_on_device));
  }
  processed_inputs.concated_sampling_params =
      inputs.concated_sampling_params.to(device_, dtype_);
  if (inputs.acc_logprob.defined()) {
    processed_inputs.acc_logprob =
        inputs.acc_logprob.to(torch::kFloat32).to(device_);
  }
  auto ret = prepare_stream_->synchronize();
}

folly::SemiFuture<bool> WorkerImpl::copy_out_blocks_async(
    ModelInputParams& input_params) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  general_threadpool_.schedule([this,
                                input_params = input_params,
                                promise = std::move(promise)]() mutable {
    c10::StreamGuard streamGuard = copy_out_stream_->set_stream_guard();
    if (input_params.async_copy_out_blocks.size() > 0) {
      const int64_t num_layers = context_.get_model_args().n_layers();
      for (int layer_id = 0; layer_id < num_layers; layer_id++) {
        auto key_cache = kv_caches_[layer_id].get_k_cache();
        auto host_k_cache = host_kv_caches_[layer_id].get_k_cache();
        auto value_cache = kv_caches_[layer_id].get_v_cache();
        auto host_v_cache = host_kv_caches_[layer_id].get_v_cache();

        for (auto block_info : input_params.async_copy_out_blocks) {
          host_k_cache[block_info.host_block_id].copy_(
              key_cache[block_info.device_block_id]);
          host_v_cache[block_info.host_block_id].copy_(
              value_cache[block_info.device_block_id]);
        }
      }

      offload_kv_blocks_to_store(input_params.async_copy_out_blocks);
    }
    auto ret = copy_out_stream_->synchronize();
    promise.setValue(ret == 0);
  });

  return future;
}

folly::SemiFuture<std::optional<ForwardOutput>> WorkerImpl::step_async(
    const BatchedForwardInputs& inputs) {
  BatchedForwardInputs batched_inputs_on_device;
  batched_inputs_on_device.micro_inputs.reserve(inputs.micro_inputs.size());

  prepare_work_before_execute(inputs, batched_inputs_on_device);

  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        inputs = std::move(batched_inputs_on_device),
                        promise = std::move(promise)]() mutable {
    // run the model on the given input in working thread
    std::vector<folly::SemiFuture<bool>> copy_futures;
    for (auto& input : inputs.micro_inputs) {
      copy_futures.push_back(
          std::move(copy_out_blocks_async(input.input_params)));
    }
    if (!enable_schedule_overlap()) {
      const auto output = this->step(inputs);
      std::for_each(copy_futures.begin(),
                    copy_futures.end(),
                    [](folly::SemiFuture<bool>& copy_future) {
                      std::move(copy_future).get();
                    });
      promise.setValue(output);
    } else {
      for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
        if (last_step_output_valid_ &&
            inputs.micro_inputs[i]
                .input_params.batch_forward_type.has_decode()) {
          // replace step i model input with true output of step i-1
          inputs.micro_inputs[i] =
              update_input_by_last_step_output(inputs.micro_inputs[i]);
        }
      }
      const auto output = this->step(inputs);
      if (output.has_value()) {
        if (is_driver() || FLAGS_enable_eplb) {
          std::unique_lock<std::mutex> lock(mtx_);
          cv_.wait(lock, [this] { return !is_recorded_; });
          update_last_step_output(output);
          is_recorded_ = true;
          cv_.notify_one();
        } else {
          update_last_step_output(output);
        }
      } else {
        if (is_driver() || FLAGS_enable_eplb) {
          std::unique_lock<std::mutex> lock(mtx_);
          cv_.wait(lock, [this] { return !is_recorded_; });
          last_step_output_valid_ = false;
          is_recorded_ = true;
          cv_.notify_one();
        } else {
          last_step_output_valid_ = false;
        }
      }
      std::for_each(copy_futures.begin(),
                    copy_futures.end(),
                    [](folly::SemiFuture<bool>& copy_future) {
                      std::move(copy_future).get();
                    });
      promise.setValue(output);
    }
  });
  return future;
}

ForwardOutput WorkerImpl::get_last_step_result() {
  ForwardOutput output;
  std::unique_lock<std::mutex> lock(mtx_);
  cv_.wait(lock, [this] { return is_recorded_; });
  if (last_step_output_valid_ || FLAGS_enable_eplb) {
    output = last_step_output_;
  }
  is_recorded_ = false;
  cv_.notify_one();
  return output;
}

folly::SemiFuture<folly::Unit> WorkerImpl::process_group_test_async() {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    this->process_group_test();
    promise.setValue();
  });
  return future;
}

// initialize model, cache manager. async call
folly::SemiFuture<bool> WorkerImpl::init_model_async(
    const std::string& model_weights_path) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, model_weights_path, promise = std::move(promise)]() mutable {
        auto status = this->init_model(model_weights_path);
        promise.setValue(status);
      });

  return future;
}

bool WorkerImpl::init_model(const std::string& model_weights_path) {
  auto model_loader = ModelLoader::create(model_weights_path);
  auto tokenizer = model_loader->tokenizer();
  CHECK(tokenizer != nullptr);

  auto args = model_loader->model_args();
  auto quant_args = model_loader->quant_args();
  torch::ScalarType dtype = util::parse_dtype(args.dtype(), device_);

  if (tokenizer->vocab_size() != args.vocab_size()) {
    // use tokenizer vocab size if model vocab size is not set
    if (args.vocab_size() <= 0) {
      LOG(WARNING)
          << "Model vocab size is not set, using tokenizer vocab size: "
          << tokenizer->vocab_size();
      args.vocab_size(tokenizer->vocab_size());
    } else {
      LOG(WARNING) << "Vocab size mismatch: tokenizer: "
                   << tokenizer->vocab_size()
                   << ", model: " << args.vocab_size();
    }
  }

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
  if (FLAGS_enable_eplb) {
    int32_t num_layers = args.n_layers() - args.first_k_dense_replace();
    int32_t num_device_experts =
        args.n_routed_experts() / context_.get_parallel_args().world_size() +
        FLAGS_redundant_experts_num;
    expert_load_data_ = torch::zeros({num_layers, num_device_experts})
                            .to(torch::kInt64)
                            .to(device_)
                            .contiguous();
  }
  return true;
}

void WorkerImpl::load_model(std::unique_ptr<ModelLoader> loader) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  model_->load_model(std::move(loader));
}

folly::SemiFuture<bool> WorkerImpl::allocate_kv_cache_async(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, &kv_cache_shape, promise = std::move(promise)]() mutable {
        const bool success = this->allocate_kv_cache(kv_cache_shape);
        promise.setValue(success);
      });
  return future;
}

folly::SemiFuture<bool> WorkerImpl::allocate_continuous_kv_cache_async(
    const std::vector<XTensor::Options>& options) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, options, promise = std::move(promise)]() mutable {
    const bool success = this->allocate_continuous_kv_cache(options);
    promise.setValue(success);
  });
  return future;
}

folly::SemiFuture<bool> WorkerImpl::pull_kv_blocks_async(
    uint64_t src_cluster_id,
    const std::string& src_addr,
    int64_t src_k_cache_id,
    int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
#if defined(USE_NPU)
  return kv_cache_transfer_->pull_kv_blocks_async(src_cluster_id,
                                                  src_addr,
                                                  src_k_cache_id,
                                                  src_v_cache_id,
                                                  src_blocks,
                                                  dst_blocks);
#endif
  return false;
}

folly::SemiFuture<uint32_t> WorkerImpl::load_kv_blocks_from_store_async(
    const std::vector<CacheBlockInfo>& cache_block_info) {
  folly::Promise<uint32_t> promise;
  auto future = promise.getSemiFuture();
  general_threadpool_.schedule(
      [this, &cache_block_info, promise = std::move(promise)]() mutable {
        if (this->kv_cache_store_ == nullptr) {
          promise.setValue(0);
          return;
        }
        promise.setValue(this->kv_cache_store_->batch_get(cache_block_info));
      });
  return future;
}

uint32_t WorkerImpl::offload_kv_blocks_to_store(
    const std::vector<CacheBlockInfo>& cache_block_info) {
  if (kv_cache_store_ == nullptr) {
    return 0;
  }
  return kv_cache_store_->batch_put(cache_block_info);
}

folly::SemiFuture<uint32_t> WorkerImpl::offload_kv_blocks_to_store_async(
    const std::vector<CacheBlockInfo>& cache_block_info) {
  folly::Promise<uint32_t> promise;
  auto future = promise.getSemiFuture();
  general_threadpool_.schedule(
      [this, &cache_block_info, promise = std::move(promise)]() mutable {
        if (this->kv_cache_store_ == nullptr) {
          promise.setValue(0);
          return;
        }
        promise.setValue(this->kv_cache_store_->batch_put(cache_block_info));
      });
  return future;
}

folly::SemiFuture<bool> WorkerImpl::allocate_kv_cache_with_transfer_async(
    uint64_t kv_cache_size,
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        kv_cache_size,
                        &kv_cache_shape,
                        promise = std::move(promise)]() mutable {
    const bool success =
        this->allocate_kv_cache_with_transfer(kv_cache_size, kv_cache_shape);
    promise.setValue(success);
  });
  return future;
}

int64_t WorkerImpl::get_active_activation_memory() {
  return DeviceMonitor::get_instance()
      .get_device_stats(device_.index())
      .active_activation_memory;
}

}  // namespace xllm
