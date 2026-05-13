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

#include "xservice_client.h"

#include <absl/strings/str_split.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <thread>
#include <unordered_map>

#include "util/env_var.h"
#include "util/hash_util.h"
#include "util/net.h"
#include "util/uuid.h"

namespace xllm {
namespace {
static std::string ETCD_MASTER_SERVICE_KEY = "XLLM:SERVICE:MASTER";
static std::string ETCD_XSERVICES_KEY_PREFIX =
    "XLLM:SERVICE:";  // all xllm_service registeration prefix
constexpr const char* kEtcdUsernameEnvVar = "ETCD_USERNAME";
constexpr const char* kEtcdPasswordEnvVar = "ETCD_PASSWORD";
constexpr int64_t kFailoverSessionLoopWaitMs = 20;
constexpr int64_t kFailoverSessionPingIntervalMs = 200;
constexpr int64_t kFailoverSessionMinRetryBackoffMs = 100;
constexpr int64_t kFailoverSessionMaxRetryBackoffMs = 5000;
std::atomic<brpc::StreamId> g_failover_session_signal_stream_id{
    brpc::INVALID_STREAM_ID};
static std::unordered_map<xllm_service::proto::InstanceType, std::string>
    ETCD_KEYS_PREFIX_MAP = {
        {xllm_service::proto::InstanceType::DEFAULT, "XLLM:DEFAULT:"},
        {xllm_service::proto::InstanceType::PREFILL, "XLLM:PREFILL:"},
        {xllm_service::proto::InstanceType::DECODE, "XLLM:DECODE:"},
        {xllm_service::proto::InstanceType::MIX, "XLLM:MIX:"},
};

std::string parse_instance_name(const std::string& name) {
  if (name.empty()) return "";
  // Validate the format of instance name
  // The format is `ip:port` currently.
  auto pos = name.find(':');
  if (pos == std::string::npos) {
    // only offer the port, we need to fill the ip address
    return xllm::net::get_local_ip_addr() + ":" + name;
  }
  return name;
}

bool check_instance_name(const std::string& name) {
  std::vector<std::string> addr = absl::StrSplit(name, ':');
  // Now only support `ip:port` format
  if (addr.size() != 2 || addr[0].empty() || addr[1].empty()) {
    LOG(ERROR)
        << "Invalid instance name format, now only support `ip:port` style.";
    return false;
  }

  return true;
}

}  // namespace

class FailoverSessionClientHandler
    : public brpc::StreamInputHandler,
      public std::enable_shared_from_this<FailoverSessionClientHandler> {
 public:
  explicit FailoverSessionClientHandler(XServiceClient* client)
      : client_(client) {}

  int on_received_messages(brpc::StreamId id,
                           butil::IOBuf* const messages[],
                           size_t size) override {
    LOG(WARNING) << "Unexpected payload on xservice failover session, "
                 << "stream_id=" << id << " message_count=" << size;
    return 0;
  }

  void on_idle_timeout(brpc::StreamId id) override {
    auto self = shared_from_this();
    client_->on_failover_session_closed(
        id, ETIMEDOUT, "xservice failover session idle timeout");
  }

  void on_closed(brpc::StreamId id) override {
    auto self = shared_from_this();
    int error_code = 0;
    std::string error_text;
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error_code = error_code_;
      error_text = error_text_;
    }
    client_->on_failover_session_closed(id, error_code, error_text);
  }

  void on_failed(brpc::StreamId id,
                 int error_code,
                 const std::string& error_text) override {
    auto self = shared_from_this();
    std::lock_guard<std::mutex> lock(error_mutex_);
    error_code_ = error_code;
    error_text_ = error_text;
  }

 private:
  XServiceClient* client_;
  std::mutex error_mutex_;
  int error_code_ = 0;
  std::string error_text_;
};

bool XServiceClient::init(const std::string& etcd_addr,
                          const std::string& instance_name,
                          const BlockManagerPool* block_manager_pool,
                          const std::string& etcd_namespace,
                          uint32_t offload_batch_size) {
  if (initialize_done_) {
    LOG(INFO) << "XServiceClient is already initialized, skipping.";
    return true;
  }

  if (etcd_addr.empty()) {
    LOG(ERROR) << "etcd_addr address is empty.";
    return false;
  }

  offload_batch_size_.store(offload_batch_size, std::memory_order_relaxed);
  instance_name_ = instance_name;
  if (incarnation_id_.empty()) {
    ShortUUID uuid;
    incarnation_id_ = uuid.random();
  }
  chan_options_.max_retry = 3;
  chan_options_.timeout_ms = FLAGS_rpc_channel_timeout_ms;

  const std::string etcd_username =
      util::get_optional_string_env(kEtcdUsernameEnvVar).value_or("");
  const std::string etcd_password =
      util::get_optional_string_env(kEtcdPasswordEnvVar).value_or("");
  const bool has_etcd_auth_user = !etcd_username.empty();
  const bool has_etcd_auth_password = !etcd_password.empty();
  if (has_etcd_auth_user != has_etcd_auth_password) {
    LOG(ERROR) << "Both " << kEtcdUsernameEnvVar << " and "
               << kEtcdPasswordEnvVar << " must be set together.";
    return false;
  }
  if (has_etcd_auth_user) {
    etcd_client_ = std::make_unique<EtcdClient>(
        etcd_addr, etcd_username, etcd_password, etcd_namespace);
  } else {
    etcd_client_ = std::make_unique<EtcdClient>(etcd_addr, etcd_namespace);
  }

  // connect master xllm_service
  while (!etcd_client_->get_master_service(ETCD_MASTER_SERVICE_KEY,
                                           &master_xservice_addr_)) {
    LOG(ERROR) << "Master service not set, wait 2s!";
    sleep(2);
  }

  if (!check_instance_name(master_xservice_addr_)) {
    LOG(FATAL) << "Invalid master service name format, now only support "
                  "`ip:port` style.";
    return false;
  }

  if (!connect_to_xservice(master_xservice_addr_)) {
    LOG(FATAL) << "Fail to initialize connection to master xservice server "
               << master_xservice_addr_;
    return false;
  }

  // Get and connect to all existing xllm_service instances.
  std::vector<std::string> all_services;
  if (etcd_client_->get_all_xservices(ETCD_XSERVICES_KEY_PREFIX,
                                      &all_services)) {
    for (const auto& service_addr : all_services) {
      if (service_addr != master_xservice_addr_ &&
          check_instance_name(service_addr)) {
        connect_to_xservice(service_addr);
      }
    }
  }

  // heartbeat thread
  heartbeat_thread_ =
      std::make_unique<std::thread>(&XServiceClient::heartbeat, this);
  reconcile_thread_ = std::make_unique<std::thread>(
      &XServiceClient::reconcile_registration_loop, this);
  failover_session_thread_ = std::make_unique<std::thread>(
      &XServiceClient::failover_session_loop, this);

  // watch master xllm_service change
  auto master_func = std::bind(&XServiceClient::handle_master_service_watch,
                               this,
                               std::placeholders::_1,
                               std::placeholders::_2);
  etcd_client_->add_watch(ETCD_MASTER_SERVICE_KEY, master_func);

  // watch all xllm_service changes
  auto xservices_func = std::bind(&XServiceClient::handle_xservices_watch,
                                  this,
                                  std::placeholders::_1,
                                  std::placeholders::_2);
  etcd_client_->add_watch(ETCD_XSERVICES_KEY_PREFIX, xservices_func);

  block_manager_pool_ = block_manager_pool;

  initialize_done_ = true;
  return true;
}

void XServiceClient::set_scheduler(Scheduler* scheduler) {
  scheduler_ = scheduler;
}

void XServiceClient::set_engine(Engine* engine) { engine_ = engine; }

XServiceClient::~XServiceClient() {
  bool expected = false;
  if (!shutdown_started_.compare_exchange_strong(expected, true)) {
    return;
  }

  std::string master_addr_to_disconnect;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    master_addr_to_disconnect = master_xservice_addr_;
  }
  LOG(INFO) << "XServiceClient cleanup start"
            << ", instance=" << instance_name_
            << ", incarnation_id=" << incarnation_id_
            << ", master_addr=" << master_addr_to_disconnect;

  exited_.store(true);
  request_failover_session_refresh();

  std::vector<brpc::StreamId> stream_ids;
  {
    std::lock_guard<std::mutex> lock(failover_session_mutex_);
    stream_ids.reserve(failover_session_handlers_.size());
    for (const auto& [stream_id, handler] : failover_session_handlers_) {
      stream_ids.push_back(stream_id);
    }
    failover_session_stream_id_ = brpc::INVALID_STREAM_ID;
    g_failover_session_signal_stream_id.store(brpc::INVALID_STREAM_ID,
                                              std::memory_order_release);
    failover_session_master_addr_.clear();
    failover_session_refresh_requested_ = false;
  }
  for (brpc::StreamId stream_id : stream_ids) {
    LOG(INFO) << "XServiceClient closing failover session"
              << ", stream_id=" << stream_id
              << ", master_addr=" << master_addr_to_disconnect;
    brpc::StreamClose(stream_id);
  }

  if (!master_addr_to_disconnect.empty()) {
    disconnect_xservice(master_addr_to_disconnect);
    LOG(INFO) << "XServiceClient disconnected master channel"
              << ", master_addr=" << master_addr_to_disconnect;
  }

  if (heartbeat_thread_ && heartbeat_thread_->joinable()) {
    heartbeat_thread_->join();
  }
  if (reconcile_thread_ && reconcile_thread_->joinable()) {
    reconcile_thread_->join();
  }
  if (failover_session_thread_ && failover_session_thread_->joinable()) {
    failover_session_thread_->join();
  }

  register_done_.store(false);
  initialize_done_ = false;
  LOG(INFO) << "XServiceClient cleanup complete"
            << ", instance=" << instance_name_
            << ", incarnation_id=" << incarnation_id_;
}

__attribute__((noinline, used)) void XServiceClient::close_failover_session() {
  const brpc::StreamId stream_id = g_failover_session_signal_stream_id.exchange(
      brpc::INVALID_STREAM_ID, std::memory_order_acq_rel);
  if (stream_id != brpc::INVALID_STREAM_ID) {
    brpc::StreamClose(stream_id);
  }
}

void XServiceClient::shutdown() { close_failover_session(); }

std::string XServiceClient::get_instance_name() { return instance_name_; }

bool XServiceClient::register_instance_with_retry(const std::string& key,
                                                  const std::string& value) {
  int retry_cnt = 0;
  while (!etcd_client_->register_instance(key, value, FLAGS_etcd_ttl)) {
    if (retry_cnt >= 30) {
      LOG(ERROR) << "Register instance failed! key: " << key;
      return false;
    }

    LOG(WARNING) << "Register instance failed, wait 2s! key: " << key;
    sleep(2);
    retry_cnt++;
  }
  return true;
}

bool XServiceClient::reconcile_registration() {
  std::string registration_key;
  std::string registration_value;
  {
    std::lock_guard<std::mutex> lock(registration_mutex_);
    if (!register_done_.load() || registration_key_.empty() ||
        registration_value_.empty()) {
      return true;
    }
    registration_key = registration_key_;
    registration_value = registration_value_;
  }

  std::string current_value;
  if (etcd_client_->get(registration_key, &current_value) &&
      !current_value.empty()) {
    return true;
  }

  LOG(WARNING) << "Detected missing instance registration in etcd, "
                  "re-registering instance: "
               << registration_key;
  return register_instance_with_retry(registration_key, registration_value);
}

void XServiceClient::reconcile_registration_loop() {
  while (!exited_.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int64_t>(FLAGS_heart_beat_interval * 1000)));
    if (!register_done_.load()) continue;

    if (!reconcile_registration()) {
      LOG(ERROR) << "Failed to reconcile instance registration in etcd.";
    }
  }
}

void XServiceClient::register_instance(const InstanceInfo& instance_info) {
  InstanceInfo registered_info = instance_info;
  registered_info.incarnation_id = incarnation_id_;
  if (registered_info.register_ts_ms == 0) {
    registered_info.register_ts_ms =
        static_cast<uint64_t>(absl::ToUnixMillis(absl::Now()));
  }

  std::string key_prefix = "";
  if (InstanceRole(registered_info.type) == InstanceRole::DEFAULT) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::DEFAULT];
  } else if (InstanceRole(registered_info.type) == InstanceRole::PREFILL) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::PREFILL];
  } else if (InstanceRole(registered_info.type) == InstanceRole::DECODE) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::DECODE];
  } else if (InstanceRole(registered_info.type) == InstanceRole::MIX) {
    key_prefix = ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::MIX];
  } else {
    LOG(ERROR) << "Unsupported instance type: " << registered_info.type;
    return;
  }

  const std::string key = key_prefix + registered_info.name;
  const std::string value = registered_info.serialize_to_json().dump();
  {
    std::lock_guard<std::mutex> lock(registration_mutex_);
    instance_name_ = registered_info.name;
    registration_key_ = key;
    registration_value_ = value;
  }

  if (!register_instance_with_retry(key, value)) {
    LOG(FATAL) << "Register instance to etcd failed!";
    return;
  }

  register_done_.store(true);
  LOG(INFO) << "Success register instance to etcd.";
}

InstanceInfo XServiceClient::get_instance_info(
    const std::string& instance_name) {
  InstanceInfo result;
  brpc::Controller cntl;
  xllm_service::proto::InstanceID req;
  xllm_service::proto::InstanceMetaInfo resp;
  req.set_name(instance_name);

  std::string master_addr;
  if (!with_master_stub(
          [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
            master_stub->GetInstanceInfo(&cntl, &req, &resp, nullptr);
          },
          &master_addr)) {
    return result;
  }

  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to get instance info from xservice server "
               << master_addr << ", error text: " << cntl.ErrorText();
    return result;
  }
  result.name = resp.name();
  result.rpc_address = resp.rpc_address();
  result.incarnation_id = resp.incarnation_id();
  result.register_ts_ms = resp.register_ts_ms();
  if (resp.type() == xllm_service::proto::InstanceType::PREFILL) {
    result.type = "PREFILL";
  } else if (resp.type() == xllm_service::proto::InstanceType::DECODE) {
    result.type = "DECODE";
  } else if (resp.type() == xllm_service::proto::InstanceType::MIX) {
    result.type = "MIX";
  } else {
    result.type = "DEFAULT";
  }
  // parse kv cache info
  for (auto& cluster_id : resp.cluster_ids()) {
    result.cluster_ids.emplace_back(cluster_id);
  }
  for (auto& addr : resp.addrs()) {
    result.addrs.emplace_back(addr);
  }
  for (auto& k_cache_id : resp.k_cache_ids()) {
    result.k_cache_ids.emplace_back(k_cache_id);
  }
  for (auto& v_cache_id : resp.v_cache_ids()) {
    result.v_cache_ids.emplace_back(v_cache_id);
  }
  result.dp_size = resp.dp_size();
  for (auto& ip : resp.device_ips()) {
    result.device_ips.emplace_back(ip);
  }
  for (auto& port : resp.ports()) {
    result.ports.emplace_back(port);
  }

  return result;
}

void XServiceClient::heartbeat() {
  KvCacheEvent event;
  while (!exited_.load()) {
    event.clear();
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int64_t>(FLAGS_heart_beat_interval * 1000)));
    if (!register_done_.load()) continue;

    if (block_manager_pool_ == nullptr || scheduler_ == nullptr) continue;

    brpc::Controller cntl;
    xllm_service::proto::HeartbeatRequest req;
    req.set_name(instance_name_);
    req.set_incarnation_id(incarnation_id_);
    if (block_manager_pool_->options().enable_prefix_cache()) {
      block_manager_pool_->get_merged_kvcache_event(&event);
      auto cache_event = req.mutable_cache_event();
      if (event.stored_cache.size()) {
        cache_event->mutable_stored_cache()->Reserve(event.stored_cache.size());
        for (auto& hash_key : event.stored_cache) {
          cache_event->add_stored_cache(hash_key.data, sizeof(hash_key.data));
        }
      }

      if (event.removed_cache.size()) {
        cache_event->mutable_removed_cache()->Reserve(
            event.removed_cache.size());
        for (auto& hash_key : event.removed_cache) {
          cache_event->add_removed_cache(hash_key.data, sizeof(hash_key.data));
        }
      }
    }

    req.mutable_load_metrics()->set_gpu_cache_usage_perc(
        block_manager_pool_->get_gpu_cache_usage_perc());

    req.mutable_load_metrics()->set_waiting_requests_num(
        scheduler_->get_waiting_requests_num());

    std::vector<int64_t> ttft;
    std::vector<int64_t> tbt;
    scheduler_->get_latency_metrics(ttft, tbt);
    if (!ttft.empty()) {
      auto max_ttft = std::max_element(ttft.begin(), ttft.end());
      req.mutable_latency_metrics()->set_recent_max_ttft(*max_ttft);
    }

    req.mutable_load_metrics()->set_offload_batch_size(
        offload_batch_size_.load(std::memory_order_relaxed));

    if (!tbt.empty()) {
      auto max_tbt = std::max_element(tbt.begin(), tbt.end());
      req.mutable_latency_metrics()->set_recent_max_tbt(*max_tbt);
    }

    // Collect XTensor info (worker free pages, model weight segments)
    if (engine_ != nullptr) {
      std::vector<size_t> worker_free_phy_pages;
      std::unordered_map<std::string, std::vector<WeightSegment>>
          model_weight_segments;
      engine_->get_xtensor_info(worker_free_phy_pages, model_weight_segments);

      auto* xtensor_info = req.mutable_xtensor_info();
      for (size_t free_pages : worker_free_phy_pages) {
        xtensor_info->add_worker_free_phy_pages(free_pages);
      }

      // Report weight segments (for non-contiguous allocation support)
      for (const auto& [model_id, segments] : model_weight_segments) {
        auto& seg_list =
            (*xtensor_info->mutable_model_weight_segments())[model_id];
        for (const auto& seg : segments) {
          auto* proto_seg = seg_list.add_segments();
          proto_seg->set_offset(seg.offset);
          proto_seg->set_size(seg.size);
        }
      }
    }

    xllm_service::proto::Status resp;
    std::string master_addr;
    if (!with_master_stub(
            [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
              master_stub->Heartbeat(&cntl, &req, &resp, nullptr);
            },
            &master_addr)) {
      continue;
    }

    if (cntl.Failed()) {
      LOG(ERROR) << "Failed to send heartbeat to master xservice "
                 << master_addr << ", error msg is: " << cntl.ErrorText();
    } else if (!resp.ok()) {
      LOG(ERROR) << "Failed to send heartbeat to master xservice "
                 << master_addr;
    }
  }
}

void XServiceClient::failover_session_loop() {
  int64_t retry_backoff_ms = kFailoverSessionMinRetryBackoffMs;
  auto next_ping_time =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(kFailoverSessionPingIntervalMs);

  while (!exited_.load()) {
    if (!register_done_.load()) {
      std::unique_lock<std::mutex> lock(failover_session_mutex_);
      failover_session_cv_.wait_for(
          lock, std::chrono::milliseconds(kFailoverSessionLoopWaitMs));
      continue;
    }

    std::string master_addr;
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      master_addr = master_xservice_addr_;
    }
    if (master_addr.empty()) {
      std::unique_lock<std::mutex> lock(failover_session_mutex_);
      failover_session_cv_.wait_for(
          lock, std::chrono::milliseconds(kFailoverSessionLoopWaitMs));
      continue;
    }

    brpc::StreamId active_stream_id = brpc::INVALID_STREAM_ID;
    std::string active_master_addr;
    bool refresh_requested = false;
    {
      std::lock_guard<std::mutex> lock(failover_session_mutex_);
      active_stream_id = failover_session_stream_id_;
      active_master_addr = failover_session_master_addr_;
      refresh_requested = failover_session_refresh_requested_;
    }

    if (active_stream_id == brpc::INVALID_STREAM_ID || refresh_requested ||
        active_master_addr != master_addr) {
      brpc::StreamId previous_stream_id = brpc::INVALID_STREAM_ID;
      if (open_failover_session_to_master(master_addr, &previous_stream_id)) {
        retry_backoff_ms = kFailoverSessionMinRetryBackoffMs;
        next_ping_time =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(kFailoverSessionPingIntervalMs);
        if (previous_stream_id != brpc::INVALID_STREAM_ID) {
          brpc::StreamClose(previous_stream_id);
        }
      } else {
        std::unique_lock<std::mutex> lock(failover_session_mutex_);
        failover_session_cv_.wait_for(
            lock, std::chrono::milliseconds(retry_backoff_ms));
        retry_backoff_ms =
            std::min(retry_backoff_ms * 2, kFailoverSessionMaxRetryBackoffMs);
        continue;
      }
    } else if (std::chrono::steady_clock::now() >= next_ping_time) {
      butil::IOBuf ping;
      ping.append("ping");
      const int write_error = brpc::StreamWrite(active_stream_id, ping);
      if (write_error != 0) {
        LOG(WARNING) << "Failover session ping failed"
                     << ", stream_id=" << active_stream_id
                     << ", master_addr=" << active_master_addr
                     << ", error_code=" << write_error;
        brpc::StreamClose(active_stream_id);
        request_failover_session_refresh();
      }
      next_ping_time =
          std::chrono::steady_clock::now() +
          std::chrono::milliseconds(kFailoverSessionPingIntervalMs);
    }

    std::unique_lock<std::mutex> lock(failover_session_mutex_);
    failover_session_cv_.wait_for(
        lock, std::chrono::milliseconds(kFailoverSessionLoopWaitMs));
  }

  std::vector<brpc::StreamId> stream_ids;
  {
    std::lock_guard<std::mutex> lock(failover_session_mutex_);
    failover_session_stream_id_ = brpc::INVALID_STREAM_ID;
    g_failover_session_signal_stream_id.store(brpc::INVALID_STREAM_ID,
                                              std::memory_order_release);
    failover_session_master_addr_.clear();
    failover_session_refresh_requested_ = false;
    stream_ids.reserve(failover_session_handlers_.size());
    for (const auto& [stream_id, handler] : failover_session_handlers_) {
      stream_ids.push_back(stream_id);
    }
  }
  for (brpc::StreamId stream_id : stream_ids) {
    brpc::StreamClose(stream_id);
  }
  for (int attempt = 0; attempt < 50; ++attempt) {
    {
      std::lock_guard<std::mutex> lock(failover_session_mutex_);
      if (failover_session_handlers_.empty()) {
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

bool XServiceClient::open_failover_session_to_master(
    const std::string& master_addr,
    brpc::StreamId* previous_stream_id) {
  if (previous_stream_id == nullptr) {
    return false;
  }
  *previous_stream_id = brpc::INVALID_STREAM_ID;

  if (!connect_to_xservice(master_addr)) {
    LOG(ERROR) << "Failed to connect to master for failover session: "
               << master_addr;
    return false;
  }

  auto handler = std::make_shared<FailoverSessionClientHandler>(this);
  brpc::Controller cntl;
  brpc::StreamOptions stream_options;
  stream_options.handler = handler.get();
  stream_options.idle_timeout_ms = -1;

  brpc::StreamId stream_id = brpc::INVALID_STREAM_ID;
  if (brpc::StreamCreate(&stream_id, cntl, &stream_options) != 0) {
    LOG(ERROR) << "Failed to create failover session stream to master: "
               << master_addr;
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(failover_session_mutex_);
    failover_session_handlers_[stream_id] = handler;
  }

  xllm_service::proto::FailoverSessionRequest req;
  req.set_name(instance_name_);
  req.set_incarnation_id(incarnation_id_);
  xllm_service::proto::Status resp;

  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto* master_stub = find_stub_locked(master_addr);
    if (master_stub == nullptr) {
      brpc::StreamClose(stream_id);
      LOG(ERROR) << "No master stub available for failover session: "
                 << master_addr;
      return false;
    }
    master_stub->OpenFailoverSession(&cntl, &req, &resp, nullptr);
  }

  if (cntl.Failed() || !resp.ok()) {
    brpc::StreamClose(stream_id);
    LOG(ERROR) << "Failed to open failover session to master " << master_addr
               << ", error="
               << (cntl.Failed() ? cntl.ErrorText() : "resp_not_ok");
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(failover_session_mutex_);
    *previous_stream_id = failover_session_stream_id_;
    failover_session_stream_id_ = stream_id;
    g_failover_session_signal_stream_id.store(stream_id,
                                              std::memory_order_release);
    failover_session_master_addr_ = master_addr;
    failover_session_refresh_requested_ = false;
  }

  LOG(INFO) << "Opened failover session to master xservice " << master_addr
            << ", instance=" << instance_name_
            << ", incarnation_id=" << incarnation_id_
            << ", stream_id=" << stream_id
            << ", replaced_stream_id=" << *previous_stream_id;
  return true;
}

void XServiceClient::on_failover_session_closed(brpc::StreamId stream_id,
                                                int error_code,
                                                const std::string& error_text) {
  bool was_active = false;
  {
    std::lock_guard<std::mutex> lock(failover_session_mutex_);
    failover_session_handlers_.erase(stream_id);
    if (failover_session_stream_id_ == stream_id) {
      failover_session_stream_id_ = brpc::INVALID_STREAM_ID;
      brpc::StreamId expected_stream_id = stream_id;
      g_failover_session_signal_stream_id.compare_exchange_strong(
          expected_stream_id,
          brpc::INVALID_STREAM_ID,
          std::memory_order_acq_rel,
          std::memory_order_acquire);
      failover_session_master_addr_.clear();
      failover_session_refresh_requested_ = true;
      was_active = true;
    } else if (failover_session_stream_id_ == brpc::INVALID_STREAM_ID) {
      failover_session_refresh_requested_ = true;
    }
  }
  LOG(INFO) << "Failover session closed"
            << ", stream_id=" << stream_id << ", was_active=" << was_active
            << ", error_code=" << error_code << ", error_text=" << error_text;
  failover_session_cv_.notify_all();
}

void XServiceClient::request_failover_session_refresh() {
  {
    std::lock_guard<std::mutex> lock(failover_session_mutex_);
    failover_session_refresh_requested_ = true;
  }
  failover_session_cv_.notify_all();
}

std::vector<std::string> XServiceClient::get_static_decode_list() {
  brpc::Controller cntl;
  xllm_service::proto::InstanceID req;
  xllm_service::proto::InstanceIDs resp;
  req.set_name(instance_name_);

  std::string master_addr;
  if (!with_master_stub(
          [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
            master_stub->GetStaticDecodeList(&cntl, &req, &resp, nullptr);
          },
          &master_addr)) {
    return {};
  }

  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to get static decode list from master xservice server "
               << master_addr << ", error text: " << cntl.ErrorText();
    return {};
  }
  return std::vector<std::string>(resp.names().begin(), resp.names().end());
}

std::vector<std::string> XServiceClient::get_static_prefill_list() {
  brpc::Controller cntl;
  xllm_service::proto::InstanceID req;
  xllm_service::proto::InstanceIDs resp;
  req.set_name(instance_name_);

  std::string master_addr;
  if (!with_master_stub(
          [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
            master_stub->GetStaticPrefillList(&cntl, &req, &resp, nullptr);
          },
          &master_addr)) {
    return {};
  }

  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to get static prefill list from master xservice server "
               << master_addr << ", error text: " << cntl.ErrorText();
    return {};
  }
  return std::vector<std::string>(resp.names().begin(), resp.names().end());
}

std::vector<std::string> XServiceClient::get_all_xservice_addrs() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<std::string> addrs;
  for (const auto& pair : xservice_stubs_) {
    addrs.push_back(pair.first);
  }
  return addrs;
}

std::vector<bool> XServiceClient::generations(
    const std::vector<RequestOutput>& outputs) {
  std::vector<bool> results(outputs.size(), false);
  std::string master_addr;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    master_addr = master_xservice_addr_;
  }

  // group requests by target xllm_service
  std::unordered_map<std::string, std::vector<size_t>> service_outputs_map;
  std::unordered_map<std::string, proto::DisaggStreamGenerations>
      service_requests_map;

  auto mark_service_failed = [&](const std::string& service_addr) {
    auto index_it = service_outputs_map.find(service_addr);
    if (index_it == service_outputs_map.end()) {
      return;
    }
    for (size_t idx : index_it->second) {
      results[idx] = false;
    }
  };

  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    std::string target_service = master_addr;
    if (!output.target_xservice_addr.empty()) {
      target_service = output.target_xservice_addr;
    }

    if (target_service.empty()) {
      LOG(ERROR) << "No target xservice address available for request_id: "
                 << output.request_id;
      continue;
    }

    service_outputs_map[target_service].push_back(i);

    // construct the request to corresponding service
    auto& gens = service_requests_map[target_service];
    proto::DisaggStreamGeneration* req = gens.mutable_gens()->Add();
    req->set_req_id(output.request_id);
    req->set_service_req_id(output.service_request_id);
    if (output.status.has_value()) {
      auto gen_status = req->mutable_gen_status();
      gen_status->set_status_code(
          static_cast<int32_t>(output.status.value().code()));
      gen_status->set_status_msg(output.status.value().message());
    }
    req->set_finished(output.finished);
    req->set_finished_on_prefill_instance(output.finished_on_prefill_instance);
    if (output.usage.has_value()) {
      proto::OutputUsage* proto_usage = req->mutable_usage();
      proto_usage->set_num_prompt_tokens(
          output.usage.value().num_prompt_tokens);
      proto_usage->set_num_generated_tokens(
          output.usage.value().num_generated_tokens);
      proto_usage->set_num_total_tokens(output.usage.value().num_total_tokens);
    }
    req->mutable_outputs()->Reserve(output.outputs.size());
    for (auto& seq_output : output.outputs) {
      auto proto_seq_out = req->mutable_outputs()->Add();
      proto_seq_out->set_index(seq_output.index);
      proto_seq_out->set_text(seq_output.text);
      if (seq_output.finish_reason.has_value()) {
        proto_seq_out->set_finish_reason(seq_output.finish_reason.value());
      } else {
        proto_seq_out->set_finish_reason("");
      }
      proto_seq_out->mutable_token_ids()->Reserve(seq_output.token_ids.size());
      for (const auto& value : seq_output.token_ids) {
        *proto_seq_out->mutable_token_ids()->Add() = value;
      }
      if (seq_output.logprobs.has_value()) {
        size_t logprobs_size = seq_output.logprobs.value().size();
        proto_seq_out->mutable_logprobs()->Reserve(logprobs_size);
        for (size_t j = 0; j < logprobs_size; ++j) {
          auto logprob = proto_seq_out->mutable_logprobs()->Add();
          proto::LogProbData* log_prob_data = logprob->mutable_log_prob_data();
          log_prob_data->set_token(seq_output.logprobs.value()[j].token);
          log_prob_data->set_token_id(seq_output.logprobs.value()[j].token_id);
          log_prob_data->set_logprob(seq_output.logprobs.value()[j].logprob);
          log_prob_data->set_finished_token(
              seq_output.logprobs.value()[j].finished_token);
          if (seq_output.logprobs.value()[j].top_logprobs.has_value()) {
            size_t top_logprobs_size =
                seq_output.logprobs.value()[j].top_logprobs.value().size();
            for (size_t k = 0; k < top_logprobs_size; ++k) {
              proto::LogProbData* top_log_prob_data =
                  logprob->mutable_top_logprobs()->Add();
              top_log_prob_data->set_token(
                  seq_output.logprobs.value()[j].top_logprobs.value()[k].token);
              top_log_prob_data->set_token_id(seq_output.logprobs.value()[j]
                                                  .top_logprobs.value()[k]
                                                  .token_id);
              top_log_prob_data->set_logprob(seq_output.logprobs.value()[j]
                                                 .top_logprobs.value()[k]
                                                 .logprob);
              top_log_prob_data->set_finished_token(
                  seq_output.logprobs.value()[j]
                      .top_logprobs.value()[k]
                      .finished_token);
            }
          }
        }
      }
    }
  }

  // Use brpc semi-synchronous RPC pattern (DoNothing + Join) instead of
  // folly thread pool. brpc::Join is bthread-aware: it yields the brpc worker
  // thread via butex_wait, avoiding the deadlock that occurs when
  // folly::SemiFuture::wait() blocks all brpc workers with a pthread futex.
  struct AsyncCallContext {
    std::string service_addr;
    brpc::Controller cntl;
    proto::StatusSet resp;
    proto::DisaggStreamGenerations gens;
    bool issued = false;
  };

  const size_t num_services = service_requests_map.size();
  std::vector<AsyncCallContext> contexts(num_services);
  std::vector<std::string> service_order;
  service_order.reserve(num_services);

  // Fire all RPCs asynchronously
  size_t idx = 0;
  for (const auto& pair : service_requests_map) {
    auto& ctx = contexts[idx++];
    ctx.service_addr = pair.first;
    ctx.gens = pair.second;
    service_order.push_back(ctx.service_addr);

    if (!connect_to_xservice(ctx.service_addr)) {
      LOG(ERROR) << "Failed to connect target xservice: " << ctx.service_addr;
      continue;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto* service_stub = find_stub_locked(ctx.service_addr);
    if (service_stub == nullptr) {
      LOG(ERROR) << "No stub available for xservice: " << ctx.service_addr;
      continue;
    }
    ctx.issued = true;
    service_stub->Generations(
        &ctx.cntl, &ctx.gens, &ctx.resp, brpc::DoNothing());
  }

  // Wait for all RPCs — bthread-aware, yields brpc worker properly
  for (auto& ctx : contexts) {
    if (ctx.issued) {
      brpc::Join(ctx.cntl.call_id());
    }
  }

  // Process results
  for (size_t i = 0; i < contexts.size(); ++i) {
    const std::string& service_addr = service_order[i];
    auto& ctx = contexts[i];
    auto index_it = service_outputs_map.find(service_addr);
    CHECK(index_it != service_outputs_map.end())
        << "No output index found for service: " << service_addr;
    const auto& indices = index_it->second;

    if (!ctx.issued || ctx.cntl.Failed()) {
      if (ctx.cntl.Failed()) {
        LOG(ERROR) << "Fail to response tokens to xservice server "
                   << service_addr << ", error text: " << ctx.cntl.ErrorText();
      }
      mark_service_failed(service_addr);
      continue;
    }

    CHECK_EQ(ctx.resp.all_status_size(), static_cast<int>(indices.size()))
        << "The size of status set is not equal to the size of outputs for "
           "service: "
        << service_addr;

    for (size_t j = 0; j < indices.size(); ++j) {
      size_t original_idx = indices[j];
      results[original_idx] = ctx.resp.all_status(j).ok();
    }
  }

  return results;
}

bool XServiceClient::connect_to_xservice(const std::string& xservice_addr) {
  if (!check_instance_name(xservice_addr)) {
    LOG(ERROR) << "Invalid xservice address format: " << xservice_addr;
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);

  // If already connected, directly return true
  if (xservice_channels_.find(xservice_addr) != xservice_channels_.end()) {
    return true;
  }

  auto channel = std::make_unique<brpc::Channel>();
  if (channel->Init(xservice_addr.c_str(), "", &chan_options_) != 0) {
    LOG(ERROR) << "Fail to initialize xservice channel to server "
               << xservice_addr;
    return false;
  }

  xservice_channels_[xservice_addr] = std::move(channel);
  xservice_stubs_[xservice_addr] =
      std::make_unique<xllm_service::proto::XllmRpcService_Stub>(
          xservice_channels_[xservice_addr].get());

  LOG(INFO) << "Successfully connected to xservice: " << xservice_addr;
  return true;
}

bool XServiceClient::with_master_stub(
    const std::function<void(xllm_service::proto::XllmRpcService_Stub*)>& fn,
    std::string* master_addr) {
  if (master_addr == nullptr) {
    return false;
  }

  // wrapper in a whole lambda function
  auto run_with_current_master_stub = [&](bool* has_master_addr) -> bool {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    *master_addr = master_xservice_addr_;
    *has_master_addr = !master_addr->empty();
    if (!*has_master_addr) {
      LOG(ERROR) << "Master xservice address is empty";
      return false;
    }

    auto* master_stub = find_stub_locked(*master_addr);
    if (master_stub == nullptr) {
      return false;
    }

    fn(master_stub);
    return true;
  };

  bool has_master_addr = false;
  if (run_with_current_master_stub(&has_master_addr)) {
    return true;
  }
  if (!has_master_addr) {
    return false;
  }

  // try re-connecting once
  if (!connect_to_xservice(*master_addr)) {
    LOG(ERROR) << "Failed to connect to master xservice: " << *master_addr;
    return false;
  }

  if (run_with_current_master_stub(&has_master_addr)) {
    return true;
  }
  if (!has_master_addr) {
    return false;
  }

  LOG(ERROR) << "No master stub available for address: " << *master_addr;
  return false;
}

xllm_service::proto::XllmRpcService_Stub* XServiceClient::find_stub_locked(
    const std::string& xservice_addr) {
  auto it = xservice_stubs_.find(xservice_addr);
  if (it == xservice_stubs_.end() || it->second == nullptr) {
    return nullptr;
  }
  return it->second.get();
}

void XServiceClient::disconnect_xservice(const std::string& xservice_addr) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (xservice_stubs_.erase(xservice_addr) > 0) {
    xservice_channels_.erase(xservice_addr);
    LOG(INFO) << "Disconnected from xservice: " << xservice_addr;

    // if master disconnected，need to update master address
    if (xservice_addr == master_xservice_addr_) {
      LOG(WARNING) << "Master xservice disconnected: " << master_xservice_addr_;
    }
  }
  lock.unlock();
  request_failover_session_refresh();
}

void XServiceClient::handle_master_service_watch(const etcd::Response& response,
                                                 const uint64_t& prefix_len) {
  if (response.events().empty() || exited_.load()) {
    return;
  }

  for (const auto& event : response.events()) {
    if (event.event_type() == etcd::Event::EventType::PUT) {
      auto new_master_addr = event.kv().as_string();

      {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (master_xservice_addr_.compare(new_master_addr) == 0) {
          continue;
        }

        LOG(INFO) << "Master service changed from " << master_xservice_addr_
                  << " to " << new_master_addr;

        master_xservice_addr_ = new_master_addr;
      }

      request_failover_session_refresh();

      if (!connect_to_xservice(new_master_addr)) {
        LOG(ERROR) << "Failed to connect to new master: " << new_master_addr;
      }
    } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      if (!master_xservice_addr_.empty()) {
        LOG(WARNING) << "Master service key deleted, clear cached master addr: "
                     << master_xservice_addr_;
        master_xservice_addr_.clear();
      }
      lock.unlock();
      request_failover_session_refresh();
    }
  }
}

void XServiceClient::handle_xservices_watch(const etcd::Response& response,
                                            const uint64_t& prefix_len) {
  if (response.events().empty() || exited_.load()) {
    return;
  }

  for (const auto& event : response.events()) {
    std::string event_key;
    std::string service_addr;
    if (event.event_type() == etcd::Event::EventType::PUT) {
      if (event.has_kv()) {
        event_key = event.kv().key().substr(prefix_len);
        service_addr = event.kv().as_string();
      }
    } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
      if (event.has_prev_kv()) {
        event_key = event.prev_kv().key().substr(prefix_len);
        service_addr = event.prev_kv().as_string();
      }
      if (service_addr.empty() && event.has_kv()) {
        if (event_key.empty()) {
          event_key = event.kv().key().substr(prefix_len);
        }
        service_addr = event.kv().as_string();
      }
    }

    if (event_key == ETCD_MASTER_SERVICE_KEY) {
      continue;
    }

    if (service_addr.empty() && !event_key.empty() &&
        event_key.rfind(ETCD_XSERVICES_KEY_PREFIX, 0) == 0) {
      service_addr = event_key.substr(ETCD_XSERVICES_KEY_PREFIX.size());
    }

    if (service_addr.empty()) {
      continue;
    }

    if (!check_instance_name(service_addr)) {
      continue;
    }

    if (event.event_type() == etcd::Event::EventType::PUT) {
      std::string master_xservice_addr;
      {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        master_xservice_addr = master_xservice_addr_;
      }

      if (service_addr != master_xservice_addr) {
        connect_to_xservice(service_addr);
      }
    } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
      disconnect_xservice(service_addr);
    }
  }
}

}  // namespace xllm
