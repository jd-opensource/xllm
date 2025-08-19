
#include "worker_impl.h"

#include <c10/core/Device.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/device_memory.h"
#include "common/device_monitor.h"
#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_loader.h"
#include "framework/parallel_state.h"
#include "framework/sampling/sampler.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/ascend/xllm_ops/replace_token.h"
#include "pytorch/adapter/utils/utils.h"
#include "util/tensor_helper.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

struct WorkerImpl::NPUStreamHelper {
  c10_npu::NPUStream H2D_memcpy_stream;
  NPUStreamHelper() : H2D_memcpy_stream(c10_npu::getNPUStreamFromPool()) {}
};

constexpr uint64_t MBUF_SIZE = 128 * 1024 * 1024;

WorkerImpl::WorkerImpl(const ParallelArgs& parallel_args,
                       const torch::Device& device,
                       const runtime::Options& options)
    : options_(options), device_(device), context_(parallel_args) {
  if (options_.enable_speculative_decode() &&
      options_.num_decoding_tokens() == 1) {
    is_spec_draft_ = true;
  }

  // first worker is the driver
  driver_ = parallel_args.rank() == 0;
  int32_t tp_size = parallel_args.world_size() / parallel_args.dp_size();
  dp_driver_ =
      parallel_args.dp_size() > 1 && parallel_args.rank() % tp_size == 0;

  int currentDevId = device.index();
  int ret = aclrtSetDevice(currentDevId);
  if (ret != 0) {
    LOG(ERROR) << "ACL set device id:" << currentDevId
               << " failed, ret:" << ret;
  }
  std::string device_name = "npu:" + std::to_string(currentDevId);
  torch_npu::init_npu(device_name);
  npu_stream_helper_ = std::make_unique<NPUStreamHelper>();
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
    key_cache = at_npu::native::npu_format_cast(
        torch::empty(kv_cache_shape[0], torch::dtype(dtype_).device(device_)),
        2);
    value_cache = at_npu::native::npu_format_cast(
        torch::empty(kv_cache_shape[1], torch::dtype(dtype_).device(device_)),
        2);
    kv_caches_.emplace_back(key_cache, value_cache);
  }

  status_ = Status::READY;
  return true;
}

bool WorkerImpl::allocate_kv_cache_with_transfer(
    uint64_t kv_cache_size,
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(kv_caches_.empty()) << "KV caches are already initialized.";

  if (FLAGS_kv_cache_transfer_type == "LlmDataDist") {
    kv_cache_transfer_ =
        std::make_shared<LlmDataDistTransfer>(options_.device_ip().value(),
                                              options_.transfer_listen_port(),
                                              options_.instance_role());

    // create a KVCache for each layer
    const int64_t num_layers = context_.get_model_args().n_layers();
    kv_caches_.reserve(num_layers);

    int32_t device_id = device_.index();
    uint64_t buf_pool_size = kv_cache_size + MBUF_SIZE;
    kv_cache_transfer_->initialize(device_id, buf_pool_size);
    kv_cache_transfer_->allocate_kv_cache(
        kv_caches_, num_layers, kv_cache_shape, dtype_);
  } else {
    kv_cache_transfer_ = std::make_unique<HcclKVCacheTransfer>(
        device_.index(), options_.transfer_listen_port());

    allocate_kv_cache(kv_cache_shape);
    kv_cache_transfer_->register_kv_cache(kv_caches_, kv_cache_shape, dtype_);
  }

  status_ = Status::READY;
  return true;
}

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

  status_ = Status::READY;
  return true;
}

void WorkerImpl::get_device_info(std::string& device_ip, uint16_t& port) {
  device_ip = options_.device_ip().value();
  port = options_.transfer_listen_port();
}

void WorkerImpl::get_cache_info(uint64_t& cluster_id,
                                std::string& addr,
                                int64_t& k_cache_id,
                                int64_t& v_cache_id) {
  kv_cache_transfer_->get_cache_info(cluster_id, addr, k_cache_id, v_cache_id);
}

bool WorkerImpl::link_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<std::string>& device_ips,
                              const std::vector<uint16_t>& ports) {
  for (int32_t i = 0; i < cluster_ids.size(); ++i) {
    if (!kv_cache_transfer_->link_cluster(
            cluster_ids[i], addrs[i], device_ips[i], ports[i])) {
      return false;
    }
  }
  return true;
}

bool WorkerImpl::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                const std::vector<std::string>& addrs,
                                const std::vector<std::string>& device_ips,
                                const std::vector<uint16_t>& ports) {
  for (int32_t i = 0; i < cluster_ids.size(); ++i) {
    if (!kv_cache_transfer_->unlink_cluster(
            cluster_ids[i], addrs[i], device_ips[i], ports[i])) {
      return false;
    }
  }
  return true;
}

std::tuple<int64_t, int64_t> WorkerImpl::estimate_kv_cache_capacity() {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  size_t torch_cache = 0;
  size_t torch_largest_block = 0;
  c10_npu::NPUCachingAllocator::emptyCache();
  c10_npu::NPUCachingAllocator::FreeDeviceCachedMemory(device_.index());
  // aclrtSynchronizeDevice();
  // get torch's cache memory size since torch_npu's emptyCache is useless
  c10_npu::NPUCachingAllocator::cacheInfo(
      device_.index(), &torch_cache, &torch_largest_block);
  const auto available_memory = DeviceMemory::available_memory(device_);
  const auto total_memory = DeviceMemory::total_memory(device_);
  DeviceMonitor::get_instance().set_total_memory(device_.index(), total_memory);
  DeviceMonitor::get_instance().set_weight_memory(
      device_.index(), total_memory - available_memory - torch_cache);
  return {available_memory + torch_cache, total_memory};
}

void WorkerImpl::process_group_test() {
  c10_npu::SetDevice(device_.index());

  // create random tensors
  const auto options = torch::dtype(torch::kHalf).device(device_);
  torch::Tensor tensor = torch::randn({10, 10}, options);
  // call allreduce
  parallel_state::reduce(tensor, context_.get_parallel_args());
  // call allgather
  parallel_state::gather(tensor, context_.get_parallel_args());
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
  if (output.value().sample_output.next_tokens.defined() || FLAGS_enable_eplb) {
    last_step_output_ = std::move(output.value());
    last_step_output_valid_ = true;
  } else {
    last_step_output_valid_ = false;
  }
}

ForwardInput WorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
#if defined(USE_A3)
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

void WorkerImpl::prepare_work_before_execute(const ForwardInput& inputs,
                                             ForwardInput& processed_inputs) {
  c10::StreamGuard streamGuard(npu_stream_helper_->H2D_memcpy_stream.unwrap());
  processed_inputs = inputs.to(device_, dtype_);

  if (!context_.get_parallel_args().mapping_data().empty()) {
    torch::Tensor token_size_per_dp_group =
        torch::tensor(processed_inputs.input_params.dp_global_token_nums,
                      torch::TensorOptions()
                          .device(torch::kCPU)
                          .dtype(torch::kInt32)
                          .pinned_memory(true));
    bool is_prefill =
        processed_inputs.input_params.global_empty_kv_cache ? true : false;
    DpEpPadding dp_ep_padding(token_size_per_dp_group,
                              context_.get_model_args().num_experts_per_tok(),
                              context_.get_parallel_args().mapping_data(),
                              device_,
                              dtype_,
                              is_prefill);
    processed_inputs.input_params.dp_ep_padding_data = dp_ep_padding.build();
    if (FLAGS_enable_eplb) {
      expert_load_data_.fill_(0);
      processed_inputs.input_params.expert_load_data = expert_load_data_;
    }
  }
  aclrtSynchronizeStream(npu_stream_helper_->H2D_memcpy_stream.stream());
}

folly::SemiFuture<std::optional<ForwardOutput>> WorkerImpl::step_async(
    const ForwardInput& inputs) {
  ForwardInput forward_inputs_on_device;
  prepare_work_before_execute(inputs, forward_inputs_on_device);

  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        inputs = forward_inputs_on_device,
                        promise = std::move(promise)]() mutable {
    // run the model on the given input in working thread
    if (!enable_schedule_overlap()) {
      const auto output = this->step(inputs);
      promise.setValue(output);
    } else {
      if (last_step_output_valid_ && !inputs.input_params.empty_kv_cache) {
        // replace step i model input with true output of step i-1
        inputs = update_input_by_last_step_output(inputs);
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
      promise.setValue(output);
    }
  });
  return future;
}

ForwardOutput WorkerImpl::get_last_step_result() {
  ForwardOutput output;
  std::unique_lock<std::mutex> lock(mtx_);
  cv_.wait(lock, [this] { return is_recorded_; });
  if (last_step_output_valid_) {
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

  // init model, create model executor
  bool status = this->init_model(dtype, args, quant_args);
  if (!status) {
    return false;
  }

  this->load_model(std::move(model_loader));

  status_ = Status::LOADED;
  if (FLAGS_enable_eplb) {
    int32_t num_layers = args.n_layers() - args.first_k_dense_replace();
    int32_t num_device_experts =
        args.n_routed_experts() / context_.get_parallel_args().world_size() + 1;
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

folly::SemiFuture<bool> WorkerImpl::pull_kv_blocks_async(
    uint64_t src_cluster_id,
    const std::string& src_addr,
    int64_t src_k_cache_id,
    int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  return kv_cache_transfer_->pull_kv_blocks_async(src_cluster_id,
                                                  src_addr,
                                                  src_k_cache_id,
                                                  src_v_cache_id,
                                                  src_blocks,
                                                  dst_blocks);
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

folly::SemiFuture<int64_t> WorkerImpl::get_active_activation_memory_async() {
  return device_monitor_.get_device_stats_async(device_.index())
      .deferValue([](const DeviceStats* stats) -> int64_t {
        return stats->active_activation_memory;
      });
}

}  // namespace xllm
