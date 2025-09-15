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

#include "dit_master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <boost/algorithm/string.hpp>
#include <csignal>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "common/metrics.h"
#include "framework/model/model_args.h"
#include "framework/request/dit_request.h"
#include "models/model_registry.h"
#include "runtime/speculative_engine.h"
#include "runtime/xservice_client.h"
#include "scheduler/scheduler_factory.h"
#include "server/xllm_server_registry.h"
#if defined(USE_NPU)
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"
#endif
#include "util/device_name_utils.h"
#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {
DiTMaster::DiTMaster(const Options& options)
    : Master(options, EngineType::DIT) {
  // TODO: init master
  CHECK(dit_engine_->init());
  LOG(INFO) << "DiT engine initialized in DiTMaster.";
  ContinuousScheduler::Options scheduler_options;
  scheduler_options.max_tokens_per_batch(options.max_tokens_per_batch())
      .max_seqs_per_batch(options.max_seqs_per_batch())
      .max_tokens_per_chunk_for_prefill(
          options.max_tokens_per_chunk_for_prefill())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_chunked_prefill(options_.enable_chunked_prefill())
      .instance_name(options_.instance_name())
      .instance_role(options_.instance_role())
      .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
      .enable_service_routing(options_.enable_service_routing());
  // scheduler_ =
  //     create_continuous_scheduler(dit_engine_.get(), scheduler_options);
  LOG(INFO) << "ContinuousScheduler created in DiTMaster.";
  InstanceInfo instance_info;
  if (options_.enable_service_routing()) {
    auto& instance_info = scheduler_->get_instance_info();
    XServiceClient::get_instance()->register_instance(instance_info);
  }
  LOG(INFO) << "Instance registered with service routing.";
  threadpool_ = std::make_unique<ThreadPool>(options.num_handling_threads());
  LOG(INFO) << "ThreadPool with " << options.num_handling_threads()
            << " threads created in DiTMaster.";
}

DiTMaster::~DiTMaster() {
  stoped_.store(true, std::memory_order_relaxed);
  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }

  // torch::cuda::empty_cache();
#if defined(USE_NPU)
  c10_npu::NPUCachingAllocator::emptyCache();
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu empty cache
#endif
}

void DiTMaster::handle_batch_request(std::vector<DiTRequestParams> sps,
                                     BatchDiTOutputCallback callback) {
  const size_t num_requests = sps.size();
  scheduler_->incr_pending_requests(num_requests);
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(sps[i]),
                   std::nullopt,
                   [i, callback](const DiTRequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void DiTMaster::handle_request(DiTRequestParams sp,
                               std::optional<Call*> call,
                               DiTOutputCallback callback) {
  LOG(INFO) << "in MM_master.cpp, into handle_request";
  auto cb = [callback = std::move(callback)](const DiTRequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };
  LOG(INFO) << "in MM_master.cpp, into handle_request with prompt";
  // add into the queue
  threadpool_->schedule(
      [this, sp = std::move(sp), callback = std::move(cb), call]() mutable {
        AUTO_COUNTER(request_handling_latency_seconds_completion);

        // remove the pending request after scheduling
        SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

        Timer timer;
        // verify the prompt
        if (!sp.verify_params(callback)) {
          return;
        }
        DiTRequestState dit_state = DiTRequestState(
            sp.input_params, sp.generation_params, callback, nullptr, call);
        auto request = std::make_shared<DiTRequest>(sp.request_id,
                                                    sp.x_request_id,
                                                    sp.x_request_time,
                                                    std::move(dit_state),
                                                    sp.service_request_id,
                                                    sp.offline,
                                                    sp.slo_ms,
                                                    sp.priority);
        if (!request) {
          return;
        }
        LOG(INFO) << "Request " << request->request_id()
                  << " created and pushing to scheduler.";
        // if (!scheduler_->add_request(request)) {
        //   CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
        //                       "No available resources to schedule request");
        // }
        LOG(INFO) << "master end handle_request";
      });
}

void DiTMaster::run() {
  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "DITMaster is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_relaxed)) {
      LOG(INFO) << "into DiTMaster::run loop";
      scheduler_->step(timeout);
    }
    LOG(INFO) << "DITMaster loop thread exiting.";
    running_.store(false, std::memory_order_relaxed);
  });
}

void DiTMaster::generate() {
  LOG(INFO) << "into DiTMaster::generate";
  DCHECK(options_.enable_schedule_overlap())
      << "Mode generate does not support schedule overlap yet.";
  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "Generate is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  scheduler_->generate();
  running_.store(false, std::memory_order_relaxed);
}

void DiTMaster::get_cache_info(std::vector<uint64_t>& cluster_ids,
                               std::vector<std::string>& addrs,
                               std::vector<int64_t>& k_cache_ids,
                               std::vector<int64_t>& v_cache_ids) {
  dit_engine_->get_cache_info(cluster_ids, addrs, k_cache_ids, v_cache_ids);
}

bool DiTMaster::link_cluster(const std::vector<uint64_t>& cluster_ids,
                             const std::vector<std::string>& addrs,
                             const std::vector<std::string>& device_ips,
                             const std::vector<uint16_t>& ports,
                             const int32_t dp_size) {
  return dit_engine_->link_cluster(
      cluster_ids, addrs, device_ips, ports, dp_size);
}

bool DiTMaster::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                               const std::vector<std::string>& addrs,
                               const std::vector<std::string>& device_ips,
                               const std::vector<uint16_t>& ports,
                               const int32_t dp_size) {
  return dit_engine_->unlink_cluster(
      cluster_ids, addrs, device_ips, ports, dp_size);
}

}  // namespace xllm