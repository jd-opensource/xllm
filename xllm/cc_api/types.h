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

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "macros.h"

namespace xllm {

using OptBool = std::optional<bool>;
using OptInt32 = std::optional<int32_t>;
using OptInt64 = std::optional<int64_t>;
using OptUInt32 = std::optional<uint32_t>;
using OptFloat = std::optional<float>;
using OptString = std::optional<std::string>;
using OptStringVec = std::optional<std::vector<std::string>>;
using OptInt32Vec = std::optional<std::vector<int32_t>>;

struct XLLM_CAPI_EXPORT XLLM_ChatMessage {
  // the role of the messages author. One of "system", "user", "assistant".
  std::string role;

  // the content of the message. null for assistant messages with function
  // calls.
  std::string content;
};

struct XLLM_CAPI_EXPORT XLLM_InitLLMOptions {
  // Whether to enable multi-head latent attention
  bool enable_mla = false;

  bool disable_chunked_prefill = false;

  bool disable_prefix_cache = false;

  // Whether to enable disaggregated prefill and decode execution
  bool enable_disagg_pd = false;

  // Whether to enable online-offline co-location in disaggregated PD mode
  bool enable_pd_ooc = false;

  // Whether to enable schedule overlap
  bool enable_schedule_overlap = false;

  // Whether to disable TTFT profiling
  bool disable_ttft_profiling = false;

  // Whether to enable forward interruption
  bool enable_forward_interruption = false;

  // Whether to enable shared memory for executing model
  bool enable_shm = false;

  bool is_local = true;

  // The KVCacheTranfer listen port
  int transfer_listen_port = 26000;

  // The number of multi-nodes
  int nnodes = 1;

  // The node rank
  int node_rank = 0;

  // Data parallel size for MLA attention
  int dp_size = 1;

  // Expert parallel size for MoE model
  int ep_size = 1;

  // Number of slots per kv cache block. Default is 128
  int block_size = 128;

  // Max gpu memory size for kv cache. Default is 0, which means cache size is
  // caculated by available memory
  int max_cache_size = 0;

  // Max number of tokens per batch
  int max_tokens_per_batch = 20480;

  // Max number of sequences per batch
  int max_seqs_per_batch = 256;

  // Max number of token per chunk in prefill stage
  int max_tokens_per_chunk_for_prefill = -1;

  // Number of speculative tokens
  int num_speculative_tokens = 0;

  // Number of threads for handling input requests
  int num_request_handling_threads = 4;

  // Expert parallel degree
  int expert_parallel_degree = 0;

  // The fraction of GPU memory to be used for model inference, including model
  // weights and kv cache
  float max_memory_utilization = 0.9;

  // The task to use the model for(e.g. generate, embed)
  std::string task = "generate";

  // NPU communication backend.(e.g. lccl, hccl). When enable dp, use hccl
  std::string communication_backend = "lccl";

  // ATB HCCL rank table file
  std::string rank_tablefile = "";

  // The role of instance(e.g. DEFAULT, PREFILL, DECODE, MIX)
  std::string instance_role = "DEFAULT";

  std::string device_ip = "";

  // The master address for multi-node distributed serving(e.g. 10.18.1.1:9999)
  std::string master_node_addr = "127.0.0.1:18899";

  // XService server address
  std::string xservice_addr = "";

  std::string instance_name = "";

  // The mode of kv cache transfer(e.g. PUSH, PULL)
  std::string kv_cache_transfer_mode = "PUSH";

  // draft hf model path to the model file
  std::optional<std::string> draft_model = std::nullopt;

  // Devices to run the draft model on, e.g. npu:0, npu:0,npu:1
  std::optional<std::string> draft_devices = std::nullopt;
};

struct XLLM_CAPI_EXPORT XLLM_RequestParams {
  // whether to include the original prompt in the response. default = true
  OptBool echo;

  OptBool offline;

  // include the log probabilities of the chosen tokens. the maximum value is 5.
  OptBool logprobs;

  // whether to ignore the end of sequence token. default = false.
  OptBool ignore_eos;

  // whether to skip special tokens in the output. default = true
  OptBool skip_special_tokens;

  // number of completions to return for each prompt. default = 1
  OptUInt32 n;

  // number of tokens to generate
  // the prompt token count + max_tokens can't exceed the model's max context
  // length.
  OptUInt32 max_tokens;

  // the number of sequence to generate server-side and returns the "best".
  // default = None
  // Results can't be streamed.
  // when used with n, best_of controls the number of candidate completions and
  // n specifies how many to return. best_of must be greater than or equal to n.
  OptUInt32 best_of;

  OptInt32 slo_ms;

  OptInt32 beam_width;

  // the number of log probabilities to include in the response, between [0,
  // 20]. default = 0
  OptInt32 top_logprobs;

  // top_k sampling cutoff, default = -1 (no cutoff)
  OptInt64 top_k;

  // top_p sampling cutoff, between [0, 1.0]. default = 1.0
  OptFloat top_p;

  // frequency penalty to reduce the likelihood of generating the same word
  // multiple times. values between [0.0, 2.0]. 0.0 means no penalty. default =
  // 0.0 Positive values penalize new tokens based on their existing frequency
  // in the text.
  OptFloat frequency_penalty;

  // presence penalty to reduce the likelihood of generating words already in
  // the prompt. values between [-2.0, 2.0]. Positive values penalize new tokens
  // based on their existing in the prompt. default = 0.0
  OptFloat presence_penalty;

  // repetition penalty to penalize new tokens based on their occurence in the
  // text. values > 1.0 encourage the model to use new tokens, while values
  // < 1.0 encourage the model to repeat tokens. default = 1.0
  OptFloat repetition_penalty;

  // temperature of the sampling, between [0, 2]. default = 0.0
  // higher value will make the ouput more random.
  OptFloat temperature;

  OptString service_request_id;

  // A unique identifier representing your end-user, which can help system to
  // monitor and detect abuse.
  OptString user;

  // up to 4 sequences where the API will stop generating further tokens.
  OptStringVec stop;

  // the list of token ids where the API will stop generating further tokens.
  OptInt32Vec stop_token_ids;
};

enum XLLM_CAPI_EXPORT XLLM_StatusCode {
  kSuccess = 0,         // Request succeeded
  kNotInitialized = 1,  // LLM instance not initialized
  kModelNotFound = 2,   // Specified model ID not loaded
  kTimeout = 3,         // Request timed out
  kInvalidRequest = 4,  // Invalid input parameters
  kInternalError = 5,   // Internal system error
};

struct XLLM_CAPI_EXPORT XLLM_Usage {
  // the number of tokens in the prompt.
  int32_t prompt_tokens;

  // the number of tokens in the generated completion.
  int32_t completion_tokens;

  // the total number of tokens used in the request (prompt + completion).
  int32_t total_tokens;
};

struct XLLM_CAPI_EXPORT XLLM_LogProbData {
  // token
  std::string token;

  // the token id.
  int32_t token_id;

  // the log probability of the token.
  float logprob;
};

struct XLLM_CAPI_EXPORT XLLM_LogProb {
  // token
  std::string token;

  // the token id.
  int32_t token_id;

  // the log probability of the token.
  float logprob;

  // the log probability of top tokens.
  std::vector<XLLM_LogProbData> top_logprobs;
};

struct XLLM_CAPI_EXPORT XLLM_Choice {
  // the index of the generated completion
  uint32_t index;

  // the generated text for completions inference
  std::optional<std::string> text;

  // the generated item for rec inference
  std::optional<uint64_t> item_id;

  // the generated message for chatcompletions inference
  std::optional<XLLM_ChatMessage> message;

  // the log probabilities of output tokens.
  std::optional<std::vector<XLLM_LogProb>> logprobs;

  // the reason of the model stoped generating tokens.
  // "stop" - the model hit a natural stop point or a provided stop sequence.
  // "length" - the maximum number of tokens specified in the request was
  // reached. "function_call" - the model called a function.
  std::string finish_reason;
};

struct XLLM_CAPI_EXPORT XLLM_Response {
  // Return code indicating request status (0 = success, non-zero = error)
  XLLM_StatusCode status_code = XLLM_StatusCode::kSuccess;

  // Optional error details (populated if status_code != kSuccess)
  std::string error_info;

  // unique id for the completion request
  std::string id;

  // the object type, which is always "text_completion".
  std::string object;

  // the unix timestamp (in seconds) of when the completion was created.
  int64_t created;

  // the model used for the completion
  std::string model;

  // list of generated completion choices for the input prompt
  std::vector<XLLM_Choice> choices;

  // usage statistics for the completion request.
  XLLM_Usage usage;
};

}  // namespace xllm
