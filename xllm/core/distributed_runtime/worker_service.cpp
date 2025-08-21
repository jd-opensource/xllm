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

#include "worker_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <vector>

#if defined(USE_NPU)
#include <c10/core/Device.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>
#endif

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/runtime/params_utils.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/forward_params.h"
#include "runtime/params_utils.h"
#include "util/timer.h"

namespace xllm {

namespace {
#if defined(USE_NPU)
void init_npu_context(const torch::Device& device) {
  int device_id = device.index();
  int ret = aclrtSetDevice(device_id);
  if (ret != 0) {
    LOG(ERROR) << "ACL set device id: " << device_id << " failed, ret:" << ret;
  }
  std::string device_name = "npu:" + std::to_string(device_id);
  torch_npu::init_npu(device_name);
}
#elif defined(USE_MLU)
// TODO(mlu): implement mlu init context
#endif
}  // namespace

#if defined(USE_NPU)
struct WorkerService::NPUStreamHelper {
  c10_npu::NPUStream D2H_memcpy_stream;
  NPUStreamHelper() : D2H_memcpy_stream(c10_npu::getNPUStreamFromPool()) {}
};
#elif defined(USE_MLU)
// TODO(mlu): implement mlu stream helper
#endif

WorkerService::WorkerService(runtime::Options options,
                             const torch::Device& device)
    : options_(options), device_(device), initialized_(false) {
#if defined(USE_NPU)
  init_npu_context(device);
  npu_stream_helper_ = std::make_unique<NPUStreamHelper>();
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu init context
#endif
}

WorkerService::WorkerService(runtime::Options options,
                             const torch::Device& device,
                             std::unique_ptr<Worker> worker)
    : options_(options),
      device_(device),
      worker_(std::move(worker)),
      initialized_(true) {
#if defined(USE_NPU)
  init_npu_context(device);
  npu_stream_helper_ = std::make_unique<NPUStreamHelper>();
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu init context
#endif
}

WorkerService::~WorkerService() = default;

void WorkerService::set_worker(std::unique_ptr<Worker> worker) {
  worker_ = std::move(worker);
  initialized_ = true;
}

void WorkerService::Hello(::google::protobuf::RpcController* controller,
                          const proto::Status* request,
                          proto::Status* response,
                          ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  if (!initialized_) {
    ctrl->SetFailed("Server is not initialized");
  } else {
    response->set_ok(true);
  }
  return;
}

void WorkerService::InitModel(::google::protobuf::RpcController* controller,
                              const proto::ModelPath* request,
                              proto::Status* response,
                              ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto model_weights_path = request->model_weights_path();
    auto init_future = worker_->init_model_async(model_weights_path);
    bool status = std::move(init_future).get();
    if (!status) {
      response->set_ok(false);
      return;
    }

    response->set_ok(true);
  });
  return;
}

void WorkerService::ProcessGroupTest(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto future = worker_->process_group_test_async();
    std::move(future).get();
    response->set_ok(true);
  });
  return;
}

void WorkerService::ProfileDeviceMemory(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::DeviceMemory* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto future = worker_->estimate_kv_cache_capacity_async();
    std::tuple<int64_t, int64_t> result = std::move(future).get();
    response->set_available_memory(std::get<0>(result));
    response->set_total_memory(std::get<1>(result));
  });
  return;
}

void WorkerService::AllocateKVCache(
    ::google::protobuf::RpcController* controller,
    const proto::KVCacheShape* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    std::vector<std::vector<int64_t>> kv_cache_shape;
    kv_cache_shape.reserve(2);
    kv_cache_shape.emplace_back(std::vector<int64_t>(
        request->key_shape().begin(), request->key_shape().end()));
    kv_cache_shape.emplace_back(std::vector<int64_t>(
        request->value_shape().begin(), request->value_shape().end()));
    auto future = worker_->allocate_kv_cache_async(kv_cache_shape);
    bool status = std::move(future).get();
    response->set_ok(status);
  });
  return;
}

void WorkerService::AllocateContinuousKVCache(
    ::google::protobuf::RpcController* controller,
    const proto::XTensorOptionsVec* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    XTensor::Options key_options;
    XTensor::Options value_options;
    key_options.num_kv_heads() = request->key_options().num_kv_heads();
    key_options.head_size() = request->key_options().head_size();
    key_options.max_context_len() = request->key_options().max_context_len();
    key_options.max_seqs_per_batch() =
        request->key_options().max_seqs_per_batch();
    value_options.num_kv_heads() = request->value_options().num_kv_heads();
    value_options.head_size() = request->value_options().head_size();
    value_options.max_context_len() =
        request->value_options().max_context_len();
    value_options.max_seqs_per_batch() =
        request->value_options().max_seqs_per_batch();
    std::vector<XTensor::Options> options_vec;
    options_vec.emplace_back(key_options);
    options_vec.emplace_back(value_options);

    auto future = worker_->allocate_continuous_kv_cache_async(options_vec);
    bool status = std::move(future).get();
    response->set_ok(status);
  });
  return;
}

void WorkerService::AllocateKVCacheWithTransfer(
    ::google::protobuf::RpcController* controller,
    const proto::AllocateKVCacheWithTransferRequest* req,
    proto::Status* resp,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    uint64_t kv_cache_size = req->kv_cache_size();
    std::vector<std::vector<int64_t>> kv_cache_shape;
    kv_cache_shape.reserve(2);
    kv_cache_shape.emplace_back(
        std::vector<int64_t>(req->kv_cache_shape().key_shape().begin(),
                             req->kv_cache_shape().key_shape().end()));
    kv_cache_shape.emplace_back(
        std::vector<int64_t>(req->kv_cache_shape().value_shape().begin(),
                             req->kv_cache_shape().value_shape().end()));
    auto future = worker_->allocate_kv_cache_with_transfer_async(
        kv_cache_size, kv_cache_shape);
    bool status = std::move(future).get();
    resp->set_ok(status);
  });
  return;
}

void WorkerService::GetCacheInfo(::google::protobuf::RpcController* controller,
                                 const proto::Empty* req,
                                 proto::CacheInfo* resp,
                                 ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    uint64_t cluster_id;
    std::string addr;
    int64_t k_cache_id;
    int64_t v_cache_id;
    worker_->get_cache_info(cluster_id, addr, k_cache_id, v_cache_id);
    resp->set_cluster_id(cluster_id);
    resp->set_addr(addr);
    resp->set_k_cache_id(k_cache_id);
    resp->set_v_cache_id(v_cache_id);
  });
  return;
}

void WorkerService::PullKVCache(::google::protobuf::RpcController* controller,
                                const proto::PullKVCacheRequest* req,
                                proto::Status* resp,
                                ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    uint64_t src_cluster_id = req->cluster_id();
    std::string addr = req->addr();
    int64_t src_k_cache_id = req->k_cache_id();
    int64_t src_v_cache_id = req->v_cache_id();
    std::vector<uint64_t> src_blocks(req->src_blocks().begin(),
                                     req->src_blocks().end());
    std::vector<uint64_t> dst_blocks(req->dst_blocks().begin(),
                                     req->dst_blocks().end());
    auto future = worker_->pull_kv_blocks_async(src_cluster_id,
                                                addr,
                                                src_k_cache_id,
                                                src_v_cache_id,
                                                src_blocks,
                                                dst_blocks);
    bool status = std::move(future).get();
    resp->set_ok(status);
  });
  return;
}

void WorkerService::LoadKVCacheFromStore(
    ::google::protobuf::RpcController* controller,
    const ::xllm::proto::CacheBlockInfos* req,
    ::xllm::proto::StoreResponse* resp,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  std::vector<CacheBlockInfo> dst_blocks;
  proto_to_cache_block_info(*req, dst_blocks);

  auto future = worker_->load_kv_blocks_from_store_async(dst_blocks);

  resp->set_success_cnt(std::move(future).get());
  return;
}

void WorkerService::GetDeviceInfo(::google::protobuf::RpcController* controller,
                                  const proto::Empty* req,
                                  proto::DeviceInfo* resp,
                                  ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    std::string device_ip;
    uint16_t listen_port;
    worker_->get_device_info(device_ip, listen_port);
    resp->set_device_ip(device_ip);
    resp->set_listen_port(listen_port);
  });
  return;
}

void WorkerService::LinkCluster(::google::protobuf::RpcController* controller,
                                const proto::ClusterInfo* req,
                                proto::Status* resp,
                                ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    std::vector<uint64_t> cluster_ids(req->cluster_ids().begin(),
                                      req->cluster_ids().end());
    std::vector<std::string> addrs(req->addrs().begin(), req->addrs().end());
    std::vector<std::string> device_ips(req->device_ips().begin(),
                                        req->device_ips().end());
    std::vector<uint16_t> ports(req->ports().begin(), req->ports().end());

    bool status = worker_->link_cluster(cluster_ids, addrs, device_ips, ports);
    resp->set_ok(status);
  });
  return;
}

void WorkerService::UnlinkCluster(::google::protobuf::RpcController* controller,
                                  const proto::ClusterInfo* req,
                                  proto::Status* resp,
                                  ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    std::vector<uint64_t> cluster_ids(req->cluster_ids().begin(),
                                      req->cluster_ids().end());
    std::vector<std::string> addrs(req->addrs().begin(), req->addrs().end());
    std::vector<std::string> device_ips(req->device_ips().begin(),
                                        req->device_ips().end());
    std::vector<uint16_t> ports(req->ports().begin(), req->ports().end());

    bool status =
        worker_->unlink_cluster(cluster_ids, addrs, device_ips, ports);
    resp->set_ok(status);
  });
  return;
}

void WorkerService::ExecuteModel(::google::protobuf::RpcController* controller,
                                 const proto::ForwardInput* pb_forward_input,
                                 proto::ForwardOutput* pb_forward_output,
                                 ::google::protobuf::Closure* done) {
  threadpool_.schedule(
      [this, controller, pb_forward_input, pb_forward_output, done]() mutable {
        brpc::ClosureGuard done_guard(done);
    // convert proto::ForwardInput to ForwardInput
#if defined(USE_NPU)
        c10_npu::SetDevice(device_.index());
#elif defined(USE_MLU)
    // TODO(mlu): implement mlu execute model
#endif
        Timer timer;
        int32_t num_sequences = pb_forward_input->num_sequences();

        // TODO: FIXME, cost to much cpu time.
        // Convert pb data to ForwardInput
        ForwardInput forward_inputs;
        proto_to_forward_input(
            pb_forward_input, forward_inputs, options_.num_decoding_tokens());

        // model output
        torch::Tensor next_tokens;
        torch::Tensor logprobs;
        torch::Tensor top_tokens;
        torch::Tensor top_logprobs;
        torch::Tensor embeddings;
        torch::Tensor expert_load_data;
        int32_t prepared_layer_id = -1;

        // execute model
        auto future = worker_->step_async(forward_inputs);

        if (!options_.enable_schedule_overlap()) {
          auto forward_outputs = std::move(future).get();
          // convert ForwardOutput to proto::ForwardOutput which contain Tokens.
          if (forward_outputs) {
            DCHECK(forward_outputs.has_value()) << "Failed to execute model";
            const auto& sample_output = forward_outputs.value().sample_output;
            expert_load_data = safe_to(
                forward_outputs.value().expert_load_data, torch::kCPU, true);
            prepared_layer_id = forward_outputs.value().prepared_layer_id;

            {
#if defined(USE_NPU)
              c10::StreamGuard streamGuard(
                  npu_stream_helper_->D2H_memcpy_stream.unwrap());
#elif defined(USE_MLU)
          // TODO(mlu): implement mlu synchronize stream
#endif
              // only driver worker (rank=0) need to fill this
              // [num_seq, ..., embed_dim] FloatTensor
              embeddings =
                  safe_to(sample_output.embeddings,
                          torch::dtype(torch::kFloat32).device(torch::kCPU),
                          true);

              // [num_seq]
              next_tokens =
                  safe_to(sample_output.next_tokens, torch::kCPU, true);
              if (next_tokens.defined()) {
                // [num_seq]
                logprobs = safe_to(sample_output.logprobs, torch::kCPU, true);
                // [num_seq, topk]
                top_tokens =
                    safe_to(sample_output.top_tokens, torch::kCPU, true);
                // [num_seq, topk]
                top_logprobs =
                    safe_to(sample_output.top_logprobs, torch::kCPU, true);
              }
#if defined(USE_NPU)
              aclrtSynchronizeStream(
                  npu_stream_helper_->D2H_memcpy_stream.stream());
#elif defined(USE_MLU)
          // TODO(mlu): implement mlu synchronize stream
#endif
            }
          }
        } else {
          if (worker_->is_driver()) {
            // construct fake output tensor
            auto options =
                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
            int32_t prefill_seq_len =
                static_cast<int32_t>(pb_forward_input->prefill_seq_len());
            next_tokens = torch::arange(
                -1, -1 * (num_sequences - prefill_seq_len + 1), -1, options);
            std::move(future).deferValue([](auto&&) {});
          }
          expert_load_data =
              torch::zeros({1, 1}).to(torch::kInt64).contiguous();
        }

        forward_output_to_proto(next_tokens,
                                logprobs,
                                top_tokens,
                                top_logprobs,
                                embeddings,
                                expert_load_data,
                                prepared_layer_id,
                                pb_forward_output);
        COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
      });
}

void WorkerService::GetLastStepResult(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* req,
    proto::ForwardOutput* pb_forward_output,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule(
      [this, controller, req, pb_forward_output, done]() mutable {
#if defined(USE_NPU)
        c10_npu::SetDevice(device_.index());
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu set device
#endif
        brpc::ClosureGuard done_guard(done);

        auto future = worker_->get_last_step_result_async();
        auto forward_outputs = std::move(future).get();
        if (forward_outputs) {
          const auto& sample_output = forward_outputs.value().sample_output;
          const auto& expert_load_data = safe_to(
              forward_outputs.value().expert_load_data, torch::kCPU, true);
          int32_t prepared_layer_id = forward_outputs.value().prepared_layer_id;
#if defined(USE_NPU)
          c10::StreamGuard streamGuard(
              npu_stream_helper_->D2H_memcpy_stream.unwrap());
#elif defined(USE_MLU)
      // TODO(mlu): implement mlu synchronize stream
#endif
          // [num_seq, ..., embed_dim]
          auto embeddings =
              safe_to(sample_output.embeddings, torch::kCPU, true);
          embeddings = safe_to(embeddings, torch::kFloat32, true);

          // [num_seq]
          const auto& next_tokens =
              safe_to(sample_output.next_tokens, torch::kCPU, true);
          if (next_tokens.defined() || FLAGS_enable_eplb) {
            // [num_seq] FloatTensor
            const auto& logprobs =
                safe_to(sample_output.logprobs, torch::kCPU, true);
            // [num_seq, topk]
            const auto& top_tokens =
                safe_to(sample_output.top_tokens, torch::kCPU, true);
            // [num_seq, topk]
            const auto& top_logprobs =
                safe_to(sample_output.top_logprobs, torch::kCPU, true);
#if defined(USE_NPU)
            aclrtSynchronizeStream(
                npu_stream_helper_->D2H_memcpy_stream.stream());
#elif defined(USE_MLU)
        // TODO(mlu): implement mlu synchronize stream
#endif

            forward_output_to_proto(next_tokens,
                                    logprobs,
                                    top_tokens,
                                    top_logprobs,
                                    embeddings,
                                    expert_load_data,
                                    prepared_layer_id,
                                    pb_forward_output);
          }
        }
      });
  return;
}

void WorkerService::GetActiveActivationMemory(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* req,
    proto::ActivationMemory* resp,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto future = worker_->get_active_activation_memory_async();
    int64_t active_activation_memory = std::move(future).get();
    resp->set_active_activation_memory(active_activation_memory);
  });
  return;
}
}  // namespace xllm
