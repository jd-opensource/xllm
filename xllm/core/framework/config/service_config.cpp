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

ServiceConfig ServiceConfig::from_flags() {
  ServiceConfig config;
  config.host(FLAGS_host)
      .port(FLAGS_port)
      .rpc_idle_timeout_s(FLAGS_rpc_idle_timeout_s)
      .rpc_channel_timeout_ms(FLAGS_rpc_channel_timeout_ms)
      .max_reconnect_count(FLAGS_max_reconnect_count)
      .num_threads(FLAGS_num_threads)
      .max_concurrent_requests(FLAGS_max_concurrent_requests)
      .num_request_handling_threads(FLAGS_num_request_handling_threads)
      .num_response_handling_threads(FLAGS_num_response_handling_threads)
      .health_check_interval_ms(FLAGS_health_check_interval_ms);
  return config;
}

ServiceConfig& ServiceConfig::get_instance() {
  static ServiceConfig config = ServiceConfig::from_flags();
  return config;
}

void ServiceConfig::reload_from_flags() {
  ServiceConfig::get_instance() = ServiceConfig::from_flags();
}

}  // namespace xllm
