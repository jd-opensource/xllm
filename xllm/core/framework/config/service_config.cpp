/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/config/service_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_json_utils.h"

DEFINE_string(host, "", "Host name for brpc server.");

DEFINE_int32(port, 8010, "Port for brpc server.");

DEFINE_int32(
    rpc_idle_timeout_s,
    -1,
    "Connection will be closed if there is no read/write operations "
    "during the last `rpc_idle_timeout_s`. -1 means wait indefinitely.");

DEFINE_int32(rpc_channel_timeout_ms,
             -1,
             "Max duration of bRPC Channel. -1 means wait indefinitely.");

DEFINE_int32(max_reconnect_count,
             40,
             "The max count for worker try to connect to server.");

DEFINE_int32(num_threads, 8, "Number of threads to process requests.");

DEFINE_int32(max_concurrent_requests,
             200,
             "Maximum number of concurrent requests the xllm service can "
             "handle. If set to 0, there is no limit.");

DEFINE_int32(num_request_handling_threads,
             4,
             "Number of threads for handling input requests.");

DEFINE_int32(num_response_handling_threads,
             4,
             "Number of threads for handling responses.");

DEFINE_int32(health_check_interval_ms,
             3000,
             "Worker health check interval in milliseconds.");

namespace xllm {

void ServiceConfig::from_flags() {
  host(FLAGS_host)
      .port(FLAGS_port)
      .rpc_idle_timeout_s(FLAGS_rpc_idle_timeout_s)
      .rpc_channel_timeout_ms(FLAGS_rpc_channel_timeout_ms)
      .max_reconnect_count(FLAGS_max_reconnect_count)
      .num_threads(FLAGS_num_threads)
      .max_concurrent_requests(FLAGS_max_concurrent_requests)
      .num_request_handling_threads(FLAGS_num_request_handling_threads)
      .num_response_handling_threads(FLAGS_num_response_handling_threads)
      .health_check_interval_ms(FLAGS_health_check_interval_ms);
}

void ServiceConfig::from_json(const JsonReader& json) {
  host(json.value_or<std::string>("host", host()))
      .port(json.value_or<int32_t>("port", port()))
      .rpc_idle_timeout_s(
          json.value_or<int32_t>("rpc_idle_timeout_s", rpc_idle_timeout_s()))
      .rpc_channel_timeout_ms(json.value_or<int32_t>("rpc_channel_timeout_ms",
                                                     rpc_channel_timeout_ms()))
      .max_reconnect_count(
          json.value_or<int32_t>("max_reconnect_count", max_reconnect_count()))
      .num_threads(json.value_or<int32_t>("num_threads", num_threads()))
      .max_concurrent_requests(json.value_or<int32_t>(
          "max_concurrent_requests", max_concurrent_requests()))
      .num_request_handling_threads(json.value_or<int32_t>(
          "num_request_handling_threads", num_request_handling_threads()))
      .num_response_handling_threads(json.value_or<int32_t>(
          "num_response_handling_threads", num_response_handling_threads()))
      .health_check_interval_ms(json.value_or<int32_t>(
          "health_check_interval_ms", health_check_interval_ms()));
}

void ServiceConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const ServiceConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, host);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, port);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, rpc_idle_timeout_s);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, rpc_channel_timeout_ms);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_reconnect_count);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, num_threads);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_concurrent_requests);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, num_request_handling_threads);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, num_response_handling_threads);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, health_check_interval_ms);
}

ServiceConfig& ServiceConfig::get_instance() {
  static ServiceConfig config;
  return config;
}

void ServiceConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
