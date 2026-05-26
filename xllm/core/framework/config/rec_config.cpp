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

#include "core/framework/config/rec_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_json_utils.h"

DEFINE_bool(
    enable_rec_fast_sampler,
    true,
    "Whether to enable RecSampler fast sampling path for Rec pipelines.");

DEFINE_bool(enable_rec_prefill_only,
            false,
            "Enable rec prefill-only mode (no decoder self-attention blocks "
            "allocation).");

DEFINE_int32(max_decode_rounds,
             0,
             "Maximum number of decode rounds for multi-step decoding. "
             "0 means disabled.");

DEFINE_bool(enable_constrained_decoding,
            false,
            "Whether to enable constrained decoding, which is used to ensure "
            "that the output meets specific format or structural requirements "
            "through pre-defined rules.");

DEFINE_bool(
    output_rec_logprobs,
    false,
    "Whether to output rec multi-round token-aligned logprobs. "
    "When enabled, missing per-token logprobs are filled with the final "
    "beam logprob.");

DEFINE_bool(enable_convert_tokens_to_item,
            false,
            "Enable token ids conversion to item id in REC/OneRec response.");

DEFINE_bool(enable_output_sku_logprobs,
            false,
            "Enable REC / OneRec token-aligned logprobs tensor output.");

DEFINE_bool(enable_extended_item_info,
            false,
            "Enable REC extended item info parsing and output tensors.");

DEFINE_int32(each_conversion_threshold,
             50,
             "Maximum number of items emitted for each REC token triplet.");

DEFINE_int32(total_conversion_threshold,
             1000,
             "Maximum total number of items emitted in one REC response.");

DEFINE_int32(request_queue_size,
             100000,
             "The request queue size of the scheduler");

DEFINE_uint32(rec_worker_max_concurrency,
              1,
              "Concurrency for rec worker parallel execution. Less than or "
              "equal to 1 means disable concurrent rec worker.");

namespace xllm {

void RecConfig::from_flags() {
  enable_rec_fast_sampler(FLAGS_enable_rec_fast_sampler)
      .enable_rec_prefill_only(FLAGS_enable_rec_prefill_only)
      .max_decode_rounds(FLAGS_max_decode_rounds)
      .enable_constrained_decoding(FLAGS_enable_constrained_decoding)
      .output_rec_logprobs(FLAGS_output_rec_logprobs)
      .enable_convert_tokens_to_item(FLAGS_enable_convert_tokens_to_item)
      .enable_output_sku_logprobs(FLAGS_enable_output_sku_logprobs)
      .enable_extended_item_info(FLAGS_enable_extended_item_info)
      .each_conversion_threshold(FLAGS_each_conversion_threshold)
      .total_conversion_threshold(FLAGS_total_conversion_threshold)
      .request_queue_size(FLAGS_request_queue_size)
      .rec_worker_max_concurrency(FLAGS_rec_worker_max_concurrency);
}

void RecConfig::from_json(const JsonReader& json) {
  enable_rec_fast_sampler(
      json.value_or<bool>("enable_rec_fast_sampler", enable_rec_fast_sampler()))
      .enable_rec_prefill_only(json.value_or<bool>("enable_rec_prefill_only",
                                                   enable_rec_prefill_only()))
      .max_decode_rounds(
          json.value_or<int32_t>("max_decode_rounds", max_decode_rounds()))
      .enable_constrained_decoding(json.value_or<bool>(
          "enable_constrained_decoding", enable_constrained_decoding()))
      .output_rec_logprobs(
          json.value_or<bool>("output_rec_logprobs", output_rec_logprobs()))
      .enable_convert_tokens_to_item(json.value_or<bool>(
          "enable_convert_tokens_to_item", enable_convert_tokens_to_item()))
      .enable_output_sku_logprobs(json.value_or<bool>(
          "enable_output_sku_logprobs", enable_output_sku_logprobs()))
      .enable_extended_item_info(json.value_or<bool>(
          "enable_extended_item_info", enable_extended_item_info()))
      .each_conversion_threshold(json.value_or<int32_t>(
          "each_conversion_threshold", each_conversion_threshold()))
      .total_conversion_threshold(json.value_or<int32_t>(
          "total_conversion_threshold", total_conversion_threshold()))
      .request_queue_size(
          json.value_or<int32_t>("request_queue_size", request_queue_size()))
      .rec_worker_max_concurrency(json.value_or<uint32_t>(
          "rec_worker_max_concurrency", rec_worker_max_concurrency()));
}

void RecConfig::append_config_json(nlohmann::ordered_json& config_json) const {
  const RecConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_rec_fast_sampler);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_rec_prefill_only);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_xattention_one_stage);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_decode_rounds);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_constrained_decoding);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, output_rec_logprobs);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_convert_tokens_to_item);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_output_sku_logprobs);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_extended_item_info);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, each_conversion_threshold);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, total_conversion_threshold);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, request_queue_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, rec_worker_max_concurrency);
}

RecConfig& RecConfig::get_instance() {
  static RecConfig config;
  return config;
}

void RecConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
