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

#include <gflags/gflags.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace xllm {

namespace {

using OptionCategory = std::pair<std::string, std::vector<std::string>>;

const OptionCategory kCommonOptions = {"COMMON OPTIONS",
                                       {"master_node_addr",
                                        "host",
                                        "port",
                                        "model",
                                        "devices",
                                        "nnodes",
                                        "node_rank",
                                        "max_tokens_per_batch",
                                        "max_seqs_per_batch",
                                        "enable_chunked_prefill",
                                        "enable_schedule_overlap",
                                        "enable_prefix_cache",
                                        "enable_shm",
                                        "enable_graph",
                                        "enable_graph_mode_decode_no_padding",
                                        "enable_prefill_piecewise_graph",
                                        "max_tokens_for_graph_mode",
                                        "communication_backend",
                                        "task"}};

const OptionCategory kCacheOptions = {"KV CACHE OPTIONS",
                                      {"block_size",
                                       "max_cache_size",
                                       "max_memory_utilization",
                                       "kv_cache_dtype"}};

const OptionCategory kMoeModelOptions = {
    "MOE MODEL OPTIONS",
    {"dp_size", "ep_size", "expert_parallel_degree"}};

const OptionCategory kDisaggregatedPrefillDecodeOptions = {
    "DISAGGREGATED PREFILL-DECODE OPTIONS",
    {"enable_disagg_pd",
     "disagg_pd_port",
     "instance_role",
     "kv_cache_transfer_mode",
     "device_ip",
     "npu_phy_id",
     "transfer_listen_port"}};

const OptionCategory kMultiStepDecodeOptions = {
    "MULTI-STEP DECODE OPTIONS",
    {"max_decode_rounds", "beam_width"}};

const OptionCategory kMtpOptions = {
    "MTP OPTIONS",
    {"draft_model", "draft_devices", "num_speculative_tokens"}};

const OptionCategory kXllmServiceOptions = {
    "XLLM-SERVICE OPTIONS",
    {"etcd_addr", "offload_batch", "rank_tablefile"}};

const OptionCategory kBeamSearchOptions = {"BEAM SEARCH OPTIONS",
                                           {"enable_beam_search_kernel",
                                            "enable_rec_fast_sampler",
                                            "enable_topk_sorted"}};

const OptionCategory kOtherOptions = {
    "OTHER OPTIONS",
    {"max_concurrent_requests",
     "model_id",
     "num_request_handling_threads",
     "num_response_handling_threads",
     "prefill_scheduling_memory_usage_threshold",
     "use_contiguous_input_buffer",
     "use_audio_in_video"}};

const std::vector<OptionCategory> kOptionCategories = {
    kCommonOptions,
    kCacheOptions,
    kMoeModelOptions,
    kDisaggregatedPrefillDecodeOptions,
    kMultiStepDecodeOptions,
    kMtpOptions,
    kXllmServiceOptions,
    kBeamSearchOptions,
    kOtherOptions};

}  // namespace

class HelpFormatter {
 private:
 public:
  static std::string generate_help() {
    std::ostringstream oss;

    oss << "USAGE: xllm --model <PATH> [OPTIONS]\n\n";

    oss << "REQUIRED OPTIONS:\n";
    oss << "  --model <PATH>: Path to the model directory. This is "
           "the only required flag.\n\n";

    oss << "HELP OPTIONS:\n";
    oss << "  -h, --help: Display this help message and exit.\n\n";

    // Print flags(options) by category
    for (const auto& [category_name, option_names] : kOptionCategories) {
      std::ostringstream category_oss;

      for (const auto& option_name : option_names) {
        google::CommandLineFlagInfo option_info;
        if (google::GetCommandLineFlagInfo(option_name.c_str(), &option_info)) {
          category_oss << "  --" << option_info.name;
          if (!option_info.description.empty()) {
            category_oss << ": " << option_info.description;
          }
          category_oss << "\n";
        }
      }

      if (!category_oss.str().empty()) {
        oss << category_name << ":\n";
        oss << category_oss.str() << "\n";
      }
    }

    oss << "For more information and all available options, visit:\n";
    oss << "  https://github.com/jd-opensource/xllm/blob/main/xllm/core/common/"
           "global_flags.cpp\n";
    oss << "Documentation: "
           "https://xllm.readthedocs.io/zh-cn/latest/cli_reference/\n";

    return oss.str();
  }

  static void print_help() { std::cout << generate_help(); }

  static void print_usage() {
    std::cout << "USAGE: xllm --model <PATH> [OPTIONS]\n";
    std::cout << "Try 'xllm --help' for more information.\n";
  }

  static void print_error(const std::string& error_msg) {
    std::cerr << "Error: " << error_msg << "\n\n";
    print_usage();
  }
};

}  // namespace xllm
