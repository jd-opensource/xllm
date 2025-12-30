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

#pragma once

#include <brpc/channel.h>

#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "disagg_pd.pb.h"
#include "framework/request/request.h"
#include "framework/tokenizer/tokenizer.h"
#include "runtime/xservice_client.h"
#include "scheduler/continuous_scheduler.h"
#include "server/xllm_server_registry.h"
#include "util/blockingconcurrentqueue.h"
#include "util/threadpool.h"

namespace xllm {

class DisaggPDScheduler : public ContinuousScheduler {
 public:
  DisaggPDScheduler(Engine* engine, const Options& options);

  virtual ~DisaggPDScheduler();

  virtual uint32_t get_waiting_requests_num() const override {
    return waiting_priority_queue_.size();
  };

  void step(const absl::Duration& timeout) override;

  bool add_request(std::shared_ptr<Request>& request) override;

  // prefill-1: for prefill send new request to decode
  virtual void dispatch_requests();
  // prefill-2: for prefill send first token to decode
  virtual void prefill_send_first_generation();
  // prefill-3: for prefill receive stream generation from decode
  virtual bool prefill_recv_generation(const RequestOutput& output);

  // decode-1: for decode recveive new request from prefill
  virtual bool decode_schedule(std::shared_ptr<Request>& request,
                               const std::string& prefill_instance_name);
  // decode-2: for decode receive first token from prefill
  virtual bool decode_recv_first_generation(
      const std::string& req_id,
      int64_t token_id,
      bool has_logprob,
      float logprob,
      std::vector<int64_t> top_tokens,
      std::vector<float> top_logprobs,
      const std::string& kv_cache_transfer_mode,
      std::vector<uint64_t> src_cluster_ids,
      std::vector<std::string> src_addrs,
      std::vector<int64_t> src_k_cache_ids,
      std::vector<int64_t> src_v_cache_ids,
      std::vector<uint64_t> src_block_ids,
      int32_t src_dp_size,
      int32_t src_dp_rank);

  // decode allocate blocks with prefix cache.
  bool try_allocate(Sequence* sequence);
  // decode-3: decode send response to prefill
  virtual bool decode_send_stream_generation(const RequestOutput& output);
  virtual std::vector<bool> decode_send_stream_generations(
      const std::vector<RequestOutput>& outputs);

  bool enable_schedule_overlap() { return options_.enable_schedule_overlap(); };

  void get_latency_metrics(std::vector<int64_t>& ttft,
                           std::vector<int64_t>& tbt);

  bool is_instance_linked(const std::string& instance_name);

  bool link_instance(const std::string& instance_name,
                     const std::vector<uint64_t>& cluster_ids,
                     const std::vector<std::string>& addrs,
                     const std::vector<std::string>& device_ips,
                     const std::vector<uint16_t>& ports,
                     const int32_t dp_size);

 protected:
  // Pre-execute prefill requests of different lengths at startup and obtain the
  // corresponding TTFT for calculating the estimated TTFT of requests.
  void profile_ttft();

  void profile_tpot();

  // check remote instance info, if not exist, get from master service
  bool check_remote_instance_info(const std::string& instance_name);

  // create rpc channel to remote instance,
  // we can get remote instance info from master service.
  proto::DisaggPDService_Stub* create_rpc_channel(
      const std::string& instance_name);

  virtual void start_rpc_server();

  // Initialize RPC server and xservice client
  // This method waits for the RPC server to be initialized and sets up the
  // xservice client connection.
  void initialize_rpc_server_and_client(const std::string& server_name);

  // Register instance information including name, RPC address, type, and cache
  // info
  void register_instance_info(const std::string& server_name, Engine* engine);

  void update_token_latency_metrics(std::vector<Sequence*>& sequences) override;

  // remote instance name(ID) -> instance info
  std::unordered_map<std::string, InstanceInfo> remote_instances_info_;

  // rpc server for prefill/decode instance
  std::unique_ptr<std::thread> rpc_server_thread_;

  // request_id -> brpc channel
  // brpc channel is connected to remote instance rpc server
  std::unordered_map<std::string, proto::DisaggPDService_Stub*>
      req_to_channel_map_;
  std::unordered_map<std::string, proto::DisaggPDService_Stub*>
      instance_channel_map_;
  std::mutex req_to_channel_map_mutex_;
  std::mutex instance_channel_map_mutex_;

  XServiceClient* xservice_client_;

  // for prefill, dispatch request to Decode instance
  std::unique_ptr<std::thread> dispatch_thread_;

  moodycamel::BlockingConcurrentQueue<std::shared_ptr<Request>>
      prefill_request_queue_;
  moodycamel::BlockingConcurrentQueue<std::shared_ptr<Request>>
      prefill_request_queue_offline_;

  // for prefill save all remote requests
  std::unordered_map<std::string, std::shared_ptr<Request>>
      remote_requests_map_;
  std::mutex remote_requests_map_mutex_;
  using RequestPriorityQueue =
      std::priority_queue<std::shared_ptr<Request>,
                          std::vector<std::shared_ptr<Request>>,
                          std::function<bool(const std::shared_ptr<Request>&,
                                             const std::shared_ptr<Request>&)>>;

  // use threadpool to handle prefill-completed request
  ThreadPool prefill_threadpool_;

  // use threadpool to handle all RequestOuputs queue
  static constexpr size_t kOutputThreadNum_ = 128;  // magic num
  size_t next_thread_idx_ = 0;
  ThreadPool output_threadpools_[kOutputThreadNum_];
  // keep the thread to handle request output
  // A request will be handled in the same thread to guarantee the token's
  // order.
  std::unordered_map<std::string, size_t> remote_requests_output_thread_map_;

  // related decode instance name(ID) list
  std::vector<std::string> decode_inst_names_;
  // TODO later
  // std::vector<std::string> updated_decode_inst_names;
  int current_decode_idx_ = 0;

  // for decode
  std::unordered_map<std::string, std::shared_ptr<Request>>
      received_request_map_;
  std::mutex received_request_map_mutex_;

  // for decode non-batch response, each request will allocate a thread to
  // handle response. keep the thread to handle request output
  std::unordered_map<std::string, size_t> received_request_output_thread_map_;

  // for decode batch response, each prefill instance will allocate a thread to
  // handle response. the requests from the same prefill will be handled in one
  // thread.
  std::unordered_map<proto::DisaggPDService_Stub*, size_t>
      remote_prefill_thread_map_;
  size_t next_prefill_thread_idx_ = 0;

  // Lock for multi-threaded read-write latency metrics
  std::vector<int64_t> recent_ttft_;
  std::vector<int64_t> recent_tbt_;
  std::mutex latency_metrics_mutex_;

  // Lock for multi-threaded read-write linked instances
  std::mutex linked_instances_mutex_;
  std::set<std::string> linked_instance_;

  std::string server_name_;
};

}  // namespace xllm
