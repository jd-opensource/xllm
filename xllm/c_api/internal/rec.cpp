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

#include "c_api/rec.h"

#include <folly/Unit.h>
#include <folly/experimental/coro/Timeout.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <pthread.h>

#include <atomic>
#include <cstring>
#include <exception>
#include <stdexcept>

#include "core/common/rec_model_utils.h"
#include "core/framework/model_loader.h"
#include "helper.h"

namespace {

const char* get_rec_pipeline_name(xllm::RecPipelineType pipeline_type) {
  switch (pipeline_type) {
    case xllm::RecPipelineType::kLlmRecDefault:
      return "LlmRecEnginePipeline";
    case xllm::RecPipelineType::kLlmRecWithMmData:
      return "LlmRecWithMmData";
    case xllm::RecPipelineType::kLlmRecMultiRoundPipeline:
      return "RecMultiRoundEnginePipeline";
    case xllm::RecPipelineType::kOneRecDefault:
      return "OneRecEnginePipeline";
    default:
      return "UnknownRecPipeline";
  }
}

void reset_pipeline_runtime_toggles() {
  FLAGS_enable_rec_fast_sampler = false;
  FLAGS_enable_prefill_piecewise_graph = false;
  FLAGS_enable_xattention_one_stage = false;
  FLAGS_enable_graph_mode_decode_no_padding = false;
  FLAGS_enable_rec_prefill_only = false;
  FLAGS_enable_constrained_decoding = false;
  FLAGS_enable_topk_sorted = false;
}

void apply_multi_round_pipeline_toggles() {
  FLAGS_enable_rec_fast_sampler = true;
  FLAGS_enable_prefill_piecewise_graph = true;
  FLAGS_enable_xattention_one_stage = false;
  FLAGS_enable_graph_mode_decode_no_padding = true;
  FLAGS_enable_topk_sorted = false;
}

void apply_onerec_pipeline_toggles(xllm::Options* options) {
  FLAGS_enable_rec_prefill_only = true;
  FLAGS_enable_constrained_decoding = true;
  FLAGS_enable_prefix_cache = false;
  FLAGS_enable_schedule_overlap = false;
  FLAGS_enable_chunked_prefill = false;

  options->enable_prefix_cache(false)
      .enable_schedule_overlap(false)
      .enable_chunked_prefill(false);

  // OneRec does not use Rec multi-round decode rounds.
  FLAGS_max_decode_rounds = 0;
}

}  // namespace

XLLM_CAPI_EXPORT XLLM_REC_Handler* xllm_rec_create(void) {
  XLLM_REC_Handler* handler = new XLLM_REC_Handler();
  CHECK(nullptr != handler);

  handler->initialized = false;

  return handler;
}

XLLM_CAPI_EXPORT void xllm_rec_destroy(XLLM_REC_Handler* handler) {
  if (!handler) return;

  handler->master.reset();
  handler->executor.reset();
  handler->model_ids.clear();
  handler->initialized = false;

  delete handler;
}

XLLM_CAPI_EXPORT void xllm_rec_init_options_default(
    XLLM_InitOptions* init_options) {
  if (nullptr == init_options) return;
  *init_options = XLLM_INIT_REC_OPTIONS_DEFAULT;
}

XLLM_CAPI_EXPORT bool xllm_rec_initialize(
    XLLM_REC_Handler* handler,
    const char* model_path,
    const char* devices,
    const XLLM_InitOptions* init_options) {
  if (!handler || !model_path || !devices) return false;

  try {
    XLLM_InitOptions xllm_init_options;
    xllm::helper::set_init_options(
        xllm::helper::BackendType::REC, init_options, &xllm_init_options);

    std::string log_dir(xllm_init_options.log_dir);
    if (!log_dir.empty()) {
      xllm::helper::init_log(xllm_init_options.log_dir);
    }

    if (!std::filesystem::exists(model_path)) {
      LOG(ERROR) << "model path[" << model_path << "] does not exist";
      return false;
    }

    xllm::Options options;
    options.model_path(model_path)
        .task_type(xllm_init_options.task)
        .devices(devices)
        .draft_model_path(xllm_init_options.draft_model)
        .draft_devices(xllm_init_options.draft_devices)
        .backend("rec")
        .block_size(xllm_init_options.block_size)
        .max_cache_size(xllm_init_options.max_cache_size)
        .max_memory_utilization(xllm_init_options.max_memory_utilization)
        .enable_prefix_cache(xllm_init_options.enable_prefix_cache)
        .max_tokens_per_batch(xllm_init_options.max_tokens_per_batch)
        .max_seqs_per_batch(xllm_init_options.max_seqs_per_batch)
        .max_tokens_per_chunk_for_prefill(
            xllm_init_options.max_tokens_per_chunk_for_prefill)
        .num_speculative_tokens(xllm_init_options.num_speculative_tokens)
        .num_request_handling_threads(
            xllm_init_options.num_request_handling_threads)
        .communication_backend(xllm_init_options.communication_backend)
        .expert_parallel_degree(xllm_init_options.expert_parallel_degree)
        .enable_chunked_prefill(xllm_init_options.enable_chunked_prefill)
        .master_node_addr(xllm_init_options.master_node_addr)
        .device_ip(xllm_init_options.device_ip)
        .transfer_listen_port(xllm_init_options.transfer_listen_port)
        .nnodes(xllm_init_options.nnodes)
        .node_rank(xllm_init_options.node_rank)
        .dp_size(xllm_init_options.dp_size)
        .ep_size(xllm_init_options.ep_size)
        .instance_name(xllm_init_options.instance_name)
        .enable_disagg_pd(xllm_init_options.enable_disagg_pd)
        .enable_schedule_overlap(xllm_init_options.enable_schedule_overlap)
        .enable_pd_ooc(xllm_init_options.enable_pd_ooc)
        .kv_cache_transfer_mode(xllm_init_options.kv_cache_transfer_mode)
        .enable_shm(xllm_init_options.enable_shm)
        .is_local(true)
        .server_idx(xllm_init_options.server_idx);

    // @TODO: Currently, gflags are configured through hard coding, which needs
    // to be improved in the future. For example, a separate gflags
    // configuration file can be provided to the so for setting gflags.
    //
    // REC so still has two configuration paths:
    // - some request/runtime code reads FLAGS_* directly
    // - master/worker construction reads xllm::Options
    //
    // The fields copied from init options below are read from FLAGS_* today.
    // beam_width/block_size/max_tokens/max_seqs are also represented in
    // Options, so duplicated values must stay aligned.
    FLAGS_beam_width = xllm_init_options.beam_width;
    FLAGS_max_decode_rounds = xllm_init_options.max_decode_rounds;
    FLAGS_max_seqs_per_batch = xllm_init_options.max_seqs_per_batch;
    FLAGS_max_tokens_per_batch = xllm_init_options.max_tokens_per_batch;
    FLAGS_block_size = xllm_init_options.block_size;
    FLAGS_enable_prefix_cache = xllm_init_options.enable_prefix_cache;
    FLAGS_enable_schedule_overlap = xllm_init_options.enable_schedule_overlap;
    FLAGS_enable_chunked_prefill = xllm_init_options.enable_chunked_prefill;

    std::unique_ptr<xllm::ModelLoader> model_loader =
        xllm::ModelLoader::create(model_path);
    const xllm::ModelArgs model_args = model_loader->model_args();
    const xllm::RecModelKind rec_model_kind =
        xllm::get_rec_model_kind(model_args.model_type());
    if (rec_model_kind == xllm::RecModelKind::kNone) {
      LOG(ERROR) << "Unsupported rec model_type: " << model_args.model_type();
      return false;
    }
    const xllm::RecPipelineType pipeline_type =
        xllm::get_rec_pipeline_type(rec_model_kind);

    // Hard-coded REC so settings. enable_graph and rec_worker_max_concurrency
    // are dual-source: runtime may read FLAGS_* while setup also needs the same
    // value in Options.
    FLAGS_enable_graph = true;
    FLAGS_rec_worker_max_concurrency = 2;

    // Pipeline-specific runtime toggles in the REC so path.
    reset_pipeline_runtime_toggles();
    switch (pipeline_type) {
      case xllm::RecPipelineType::kLlmRecMultiRoundPipeline:
        apply_multi_round_pipeline_toggles();
        break;
      case xllm::RecPipelineType::kOneRecDefault:
        apply_onerec_pipeline_toggles(&options);
        break;
      case xllm::RecPipelineType::kLlmRecDefault:
      case xllm::RecPipelineType::kLlmRecWithMmData:
        break;
      default:
        LOG(ERROR) << "Unsupported rec pipeline type: "
                   << static_cast<int>(pipeline_type);
        return false;
    }

    // Keep dual-source settings aligned with the FLAGS_* values above.
    options.enable_graph(FLAGS_enable_graph)
        .beam_width(FLAGS_beam_width)
        .rec_worker_max_concurrency(FLAGS_rec_worker_max_concurrency);

    LOG(INFO) << "REC C API selected pipeline="
              << get_rec_pipeline_name(pipeline_type)
              << ", model_type=" << model_args.model_type()
              << ", enable_rec_prefill_only=" << FLAGS_enable_rec_prefill_only
              << ", enable_constrained_decoding="
              << FLAGS_enable_constrained_decoding
              << ", enable_prefix_cache=" << FLAGS_enable_prefix_cache
              << ", enable_schedule_overlap=" << FLAGS_enable_schedule_overlap
              << ", enable_chunked_prefill=" << FLAGS_enable_chunked_prefill
              << ", enable_rec_fast_sampler=" << FLAGS_enable_rec_fast_sampler
              << ", max_decode_rounds=" << FLAGS_max_decode_rounds;

#if !defined(USE_NPU)
    FLAGS_enable_block_copy_kernel = false;
#endif

    handler->master = std::make_unique<xllm::RecMaster>(options);
    handler->master->run();

    size_t cpu_cores = std::thread::hardware_concurrency();
    size_t thread_num = std::clamp((cpu_cores == 0) ? 8 : cpu_cores / 2,
                                   static_cast<size_t>(4),
                                   static_cast<size_t>(16));
    handler->executor =
        std::make_unique<folly::CPUThreadPoolExecutor>(thread_num);

    std::filesystem::path model_path_fs =
        std::filesystem::path(model_path).lexically_normal();
    std::string model_id;
    if (model_path_fs.has_filename()) {
      model_id = model_path_fs.filename().string();
    } else if (!model_path_fs.empty()) {
      model_id = model_path_fs.string();
    } else {
      model_id = "default";
    }
    handler->model_ids.clear();
    handler->model_ids.emplace_back(model_id);

    handler->initialized = true;

    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "LLM initialization failed: " << e.what();
  }

  handler->master.reset();
  handler->executor.reset();
  handler->model_ids.clear();
  handler->initialized = false;

  return false;
}

XLLM_CAPI_EXPORT void xllm_rec_request_params_default(
    XLLM_RequestParams* request_params) {
  if (nullptr == request_params) return;
  *request_params = XLLM_REC_REQUEST_PARAMS_DEFAULT;
}

XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_text_completions(
    XLLM_REC_Handler* handler,
    const char* model_id,
    const char* prompt,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !prompt ||
      *prompt == '\0') {
    return xllm::helper::build_error_response(
        "", XLLM_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  return xllm::helper::handle_inference_request(
      handler,
      xllm::helper::InferenceType::REC_COMPLETIONS,
      model_id,
      prompt,
      nullptr,
      timeout_ms,
      request_params);
}

XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_token_completions(
    XLLM_REC_Handler* handler,
    const char* model_id,
    const int32_t* token_ids,
    size_t token_size,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !token_ids ||
      token_size == 0) {
    return xllm::helper::build_error_response(
        "", XLLM_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  std::vector<int> token_ids_vec;
  for (int i = 0; i < token_size; i++) {
    token_ids_vec.push_back(token_ids[i]);
  }

  return xllm::helper::handle_inference_request(
      handler,
      xllm::helper::InferenceType::REC_COMPLETIONS,
      model_id,
      token_ids_vec,
      nullptr,
      timeout_ms,
      request_params);
}

XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_multimodal_completions(
    XLLM_REC_Handler* handler,
    const char* model_id,
    const int32_t* token_ids,
    size_t token_size,
    const XLLM_MM_Data* mm_data,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !token_ids ||
      token_size == 0) {
    return xllm::helper::build_error_response(
        "", XLLM_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  if (!mm_data) {
    return xllm_rec_token_completions(
        handler, model_id, token_ids, token_size, timeout_ms, request_params);
  }

  xllm::MMData internal_mm_data;
  try {
    bool ret = xllm::helper::convert_xllm_mm_data_to_internal(mm_data,
                                                              internal_mm_data);
    if (!ret) {
      return xllm::helper::build_error_response(
          "", XLLM_StatusCode::kInternalError, "Fail in mm_data conversion");
    }
  } catch (const std::exception& e) {
    return xllm::helper::build_error_response(
        "",
        XLLM_StatusCode::kInternalError,
        "Critical error in mm_data conversion: " + std::string(e.what()));
  }

  std::vector<int> token_ids_vec;
  for (int i = 0; i < token_size; i++) {
    token_ids_vec.push_back(token_ids[i]);
  }

  return xllm::helper::handle_inference_request(
      handler,
      xllm::helper::InferenceType::REC_COMPLETIONS,
      model_id,
      token_ids_vec,
      static_cast<void*>(&internal_mm_data),
      timeout_ms,
      request_params);
}

XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_chat_completions(
    XLLM_REC_Handler* handler,
    const char* model_id,
    const XLLM_ChatMessage* messages,
    size_t messages_count,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !messages ||
      messages_count == 0) {
    return xllm::helper::build_error_response(
        "", XLLM_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  std::vector<xllm::Message> xllm_messages;
  xllm_messages.reserve(messages_count);
  for (int i = 0; i < messages_count; i++) {
    xllm_messages.emplace_back(messages[i].role, messages[i].content);
  }

  return xllm::helper::handle_inference_request(
      handler,
      xllm::helper::InferenceType::REC_CHAT_COMPLETIONS,
      model_id,
      xllm_messages,
      nullptr,
      timeout_ms,
      request_params);
}

XLLM_CAPI_EXPORT void xllm_rec_free_response(XLLM_Response* resp) {
  return xllm::helper::xllm_free_response(resp);
}
