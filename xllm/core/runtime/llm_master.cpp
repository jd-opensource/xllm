#include "llm_master.h"

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

#include "common/metrics.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "models/model_registry.h"
#include "runtime/speculative_engine.h"
#include "runtime/xservice_client.h"
#include "scheduler/scheduler_factory.h"
#include "server/xllm_server_registry.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"
#include "util/device_name_utils.h"
#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {
volatile bool LLMAssistantMaster::running_ = false;

LLMMaster::LLMMaster(const Options& options)
    : Master(options,
             options.draft_model_path().value_or("").empty()
                 ? EngineType::LLM
                 : EngineType::SSM) {
  CHECK(engine_->init());

  model_args_ = engine_->model_args();

  bool enable_decode_response_to_service = false;
  if (options_.enable_service_routing()) {
    XServiceClient* xservice_client = XServiceClient::get_instance();
    if (!xservice_client->init(options_.etcd_addr().value_or(""),
                               options_.xservice_addr().value_or(""),
                               options_.instance_name().value_or(""),
                               engine_->block_manager_pool())) {
      LOG(FATAL) << "XServiceClient init fail!";
      return;
    }
    auto service_config = xservice_client->get_config();
    enable_decode_response_to_service =
        service_config.enable_decode_response_to_service;
  }

  ContinuousScheduler::Options scheduler_options;
  scheduler_options.max_tokens_per_batch(options_.max_tokens_per_batch())
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .max_tokens_per_chunk_for_prefill(
          options_.max_tokens_per_chunk_for_prefill())
      .num_speculative_tokens(options_.num_speculative_tokens())
      .dp_size(options_.dp_size())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_schedule_overlap(options_.enable_schedule_overlap())
      .enable_chunked_prefill(options_.enable_chunked_prefill())
      .instance_role(options_.instance_role())
      .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
      .enable_service_routing(options_.enable_service_routing())
      .enable_decode_response_to_service(enable_decode_response_to_service);
  scheduler_ = create_continuous_scheduler(engine_.get(), scheduler_options);

  if (options_.enable_service_routing()) {
    InstanceInfo instance_info;

    instance_info.name = options_.instance_name().value_or("");
    switch (options_.instance_role()) {
      case InstanceRole::DEFAULT:
        instance_info.rpc_address = "";
        instance_info.type = "default";
        break;

      case InstanceRole::PREFILL:
        instance_info.rpc_address = ServerRegistry::get_instance()
                                        .get_server("DisaggPrefillServer")
                                        ->listen_address();
        instance_info.type = "prefill";
        engine_->get_cache_info(instance_info.cluster_ids,
                                instance_info.addrs,
                                instance_info.k_cache_ids,
                                instance_info.v_cache_ids);
        break;

      case InstanceRole::DECODE:
        instance_info.rpc_address = ServerRegistry::get_instance()
                                        .get_server("DisaggDecodeServer")
                                        ->listen_address();
        instance_info.type = "decode";
        engine_->get_cache_info(instance_info.cluster_ids,
                                instance_info.addrs,
                                instance_info.k_cache_ids,
                                instance_info.v_cache_ids);
        break;

      default:
        LOG(FATAL) << "Unrecognized InstanceRole:"
                   << options_.instance_role().to_string();
        return;
    }

    instance_info.dp_size = options.dp_size();

    XServiceClient::get_instance()->register_instance(instance_info);
  }

  // construct chat template
  chat_template_ =
      std::make_unique<JinjaChatTemplate>(engine_->tokenizer_args());

  tokenizer_ = engine_->tokenizer()->clone();
  threadpool_ = std::make_unique<ThreadPool>(options_.num_handling_threads());
}

LLMMaster::~LLMMaster() {
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

void LLMMaster::handle_batch_request(std::vector<std::string> prompts,
                                     std::vector<RequestParams> sps,
                                     BatchOutputCallback callback) {
  CHECK(prompts.size() == sps.size() || sps.size() == 1)
      << "Number of prompts and sampling parameters should be the same";

  const size_t num_requests = prompts.size();
  scheduler_->incr_pending_requests(num_requests);
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(prompts[i]),
                   std::nullopt,
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void LLMMaster::handle_batch_request(
    std::vector<std::vector<Message>> conversations,
    std::vector<RequestParams> sps,
    BatchOutputCallback callback) {
  CHECK(conversations.size() == sps.size() || sps.size() == 1)
      << "Number of conversations and sampling parameters should be the same";

  const size_t num_requests = conversations.size();
  scheduler_->incr_pending_requests(num_requests);
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(conversations[i]),
                   std::nullopt,
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void LLMMaster::handle_request(std::string prompt,
                               std::optional<std::vector<int>> prompt_tokens,
                               RequestParams sp,
                               OutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback)](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };
  // add into the queue
  threadpool_->schedule([this,
                         prompt = std::move(prompt),
                         prompt_token = std::move(prompt_tokens),
                         sp = std::move(sp),
                         callback = std::move(cb)]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_completion);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    Timer timer;
    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    auto request = generate_request(
        std::move(prompt), std::move(prompt_token), sp, callback);
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

void LLMMaster::handle_request(std::vector<Message> messages,
                               std::optional<std::vector<int>> prompt_tokens,
                               RequestParams sp,
                               OutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback)](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };
  // add into the queue
  threadpool_->schedule([this,
                         messages = std::move(messages),
                         prompt_token = std::move(prompt_tokens),
                         sp = std::move(sp),
                         callback = std::move(cb)]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_chat);
    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    auto request =
        generate_request(messages, std::move(prompt_token), sp, callback);
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

void LLMMaster::run() {
  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "LLMMaster is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_relaxed)) {
      scheduler_->step(timeout);
    }
    running_.store(false, std::memory_order_relaxed);
  });
}

void LLMMaster::generate() {
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

std::shared_ptr<Request> LLMMaster::generate_request(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    const RequestParams& sp,
    OutputCallback callback) {
  if (prompt.empty()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is empty");
    return nullptr;
  }

  // encode the prompt
  Timer timer;
  std::vector<int> local_prompt_tokens;

  if (prompt_tokens.has_value()) {
    local_prompt_tokens = std::move(prompt_tokens.value());
  } else {
    if (!get_tls_tokenizer()->encode(prompt, &local_prompt_tokens)) {
      LOG(ERROR) << "Failed to encode prompt: " << prompt;
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Failed to encode prompt");
      return nullptr;
    }
  }

  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());

  int32_t max_context_len = model_args_.max_position_embeddings();
  if (!options_.enable_chunked_prefill()) {
    max_context_len =
        std::min(max_context_len, options_.max_tokens_per_batch());
  }
  if (local_prompt_tokens.size() >= max_context_len) {
    LOG(ERROR) << "Prompt is too long: " << local_prompt_tokens.size();
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is too long");
    return nullptr;
  }

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 5120;
    max_tokens = kDefaultMaxTokens;
  }

  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens
  size_t capacity = local_prompt_tokens.size() + max_tokens +
                    options_.num_speculative_tokens() + /*bouns_token*/ 1;
  if (options_.enable_schedule_overlap()) {
    capacity += options_.num_speculative_tokens() + 1;
  }
  const size_t best_of = sp.best_of.value_or(sp.n);

  RequestSamplingParam sampling_param;
  sampling_param.frequency_penalty = sp.frequency_penalty;
  sampling_param.presence_penalty = sp.presence_penalty;
  sampling_param.repetition_penalty = sp.repetition_penalty;
  sampling_param.temperature = sp.temperature;
  sampling_param.top_p = sp.top_p;
  sampling_param.top_k = sp.top_k;
  sampling_param.logprobs = sp.logprobs;
  sampling_param.top_logprobs = sp.top_logprobs;
  sampling_param.is_embeddings = sp.is_embeddings;
  if (best_of > sp.n) {
    // enable logprobs for best_of to generate sequence logprob
    sampling_param.logprobs = true;
  }
  // sampling_param.do_sample = sp.do_sample;

  std::unordered_set<int32_t> stop_tokens;
  if (sp.stop_token_ids.has_value()) {
    const auto& stop_token_ids = sp.stop_token_ids.value();
    stop_tokens.insert(stop_token_ids.begin(), stop_token_ids.end());
  } else {
    stop_tokens = model_args_.stop_token_ids();
  }
  std::vector<std::vector<int32_t>> stop_sequences;
  if (sp.stop.has_value()) {
    for (const auto& s : sp.stop.value()) {
      std::vector<int> tmp_tokens;
      if (!get_tls_tokenizer()->encode(s, &tmp_tokens)) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Failed to encode stop sequence");
        LOG(ERROR) << "Failed to encode stop sequence: " << s;
        return nullptr;
      }
      stop_sequences.push_back(std::move(tmp_tokens));
    }
  }

  StoppingChecker stopping_checker(
      max_tokens,
      max_context_len - options_.num_speculative_tokens(),
      model_args_.eos_token_id(),
      sp.ignore_eos,
      std::move(stop_tokens),
      std::move(stop_sequences));

  auto finish_reason =
      stopping_checker.check(local_prompt_tokens, local_prompt_tokens.size());
  if (finish_reason != FinishReason::NONE) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Invalid Prompt");
    LOG(ERROR) << "Invalid Prompt EndWith Token_ID:"
               << local_prompt_tokens[local_prompt_tokens.size() - 1];
    return nullptr;
  }

  bool stream = sp.streaming;
  // results cannot be streamed when best_of != n
  if (best_of != sp.n) {
    stream = false;
  }

  RequestState req_state(std::move(prompt),
                         std::move(local_prompt_tokens),
                         std::move(sampling_param),
                         std::move(stopping_checker),
                         capacity,
                         sp.n,
                         best_of,
                         sp.logprobs,
                         stream,
                         sp.echo,
                         sp.skip_special_tokens,
                         options_.enable_schedule_overlap(),
                         callback,
                         nullptr,
                         sp.decode_address);

  auto request = std::make_shared<Request>(sp.request_id,
                                           sp.x_request_id,
                                           sp.x_request_time,
                                           std::move(req_state),
                                           sp.service_request_id);

  // add one sequence, rest will be added by scheduler
  return request;
}

std::shared_ptr<Request> LLMMaster::generate_request(
    const std::vector<Message>& messages,
    std::optional<std::vector<int>> prompt_tokens,
    const RequestParams& sp,
    OutputCallback callback) {
  Timer timer;
  std::optional<std::string> prompt;
  if (sp.has_tools()) {
    prompt = chat_template_->apply(messages, sp.tools);
  } else {
    prompt = chat_template_->apply(messages);
  }

  if (!prompt.has_value()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to construct prompt from messages");
    LOG(ERROR) << "Failed to construct prompt from messages";
    return nullptr;
  }
  COUNTER_ADD(chat_template_latency_seconds, timer.elapsed_seconds());

  return generate_request(
      std::move(prompt.value()), std::move(prompt_tokens), sp, callback);
}

void LLMMaster::get_cache_info(std::vector<uint64_t>& cluster_ids,
                               std::vector<std::string>& addrs,
                               std::vector<int64_t>& k_cache_ids,
                               std::vector<int64_t>& v_cache_ids) {
  engine_->get_cache_info(cluster_ids, addrs, k_cache_ids, v_cache_ids);
}

bool LLMMaster::link_cluster(const std::vector<uint64_t>& cluster_ids,
                             const std::vector<std::string>& addrs,
                             const std::vector<std::string>& device_ips,
                             const std::vector<uint16_t>& ports,
                             const int32_t dp_size) {
  return engine_->link_cluster(cluster_ids, addrs, device_ips, ports, dp_size);
}

bool LLMMaster::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                               const std::vector<std::string>& addrs,
                               const std::vector<std::string>& device_ips,
                               const std::vector<uint16_t>& ports,
                               const int32_t dp_size) {
  return engine_->unlink_cluster(
      cluster_ids, addrs, device_ips, ports, dp_size);
}

Tokenizer* LLMMaster::get_tls_tokenizer() {
  thread_local std::unique_ptr<Tokenizer> tls_tokenizer(tokenizer_->clone());
  return tls_tokenizer.get();
}

LLMAssistantMaster::LLMAssistantMaster(const Options& options)
    : Master(options,
             options.draft_model_path().value_or("").empty()
                 ? EngineType::LLM
                 : EngineType::SSM) {
  // setup process workers
  auto master_node_addr = options_.master_node_addr().value_or("");
  // TODO: support local unix domain socket later.
  if (master_node_addr.empty()) {
    LOG(FATAL)
        << "MultiNodeEngine required master_node_addr, current value is empty.";
    return;
  }

  running_ = true;
}

void LLMAssistantMaster::run() {
  signal(SIGINT, LLMAssistantMaster::handle_signal);
  signal(SIGTERM, LLMAssistantMaster::handle_signal);

  while (running_) {
    sleep(5);
  }
}

}  // namespace xllm
