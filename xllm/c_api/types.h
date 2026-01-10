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

#ifndef XLLM_C_TYPES_H
#define XLLM_C_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Maximum length for meta string fields (includes '\0' terminator)
#define XLLM_META_STRING_FIELD_MAX_LEN 128

// Export Macro Definition
#ifndef XLLM_CAPI_EXPORT
#define XLLM_CAPI_EXPORT __attribute__((visibility("default")))
#endif

// Core Struct & Enum Definitions

/**
 * @brief Configuration options for initializing an LLM instance
 * @note All string fields are fixed-length arrays. Default values are defined
 * in macros. Empty string indicates disable/use default value.
 */
typedef struct XLLM_CAPI_EXPORT XLLM_InitOptions {
  /** Whether to enable multi-head latent attention */
  bool enable_mla;

  /** Whether to enable chunked prefill for inference */
  bool enable_chunked_prefill;

  /** Whether to enable prefix cache optimization */
  bool enable_prefix_cache;

  /** Whether to enable disaggregated prefill and decode execution */
  bool enable_disagg_pd;

  /** Whether to enable online-offline co-location in disaggregated PD mode */
  bool enable_pd_ooc;

  /** Whether to enable schedule overlap for parallel execution */
  bool enable_schedule_overlap;

  /** Whether to enable shared memory for model execution */
  bool enable_shm;

  /** KVCache transfer listen port */
  uint32_t transfer_listen_port;

  /** Number of multi-nodes in distributed deployment */
  uint32_t nnodes;

  /** Node rank in distributed deployment */
  uint32_t node_rank;

  /** Data parallel size for MLA attention */
  uint32_t dp_size;

  /** Expert parallel size for MoE model */
  uint32_t ep_size;

  /** Number of slots per kv cache block */
  uint32_t block_size;

  /** Max GPU memory size for kv cache (0 = auto-calculate available memory) */
  uint32_t max_cache_size;

  /** Max number of tokens per batch */
  uint32_t max_tokens_per_batch;

  /** Max number of sequences per batch */
  uint32_t max_seqs_per_batch;

  /** Max number of token per chunk in prefill stage */
  uint32_t max_tokens_per_chunk_for_prefill;

  /** Number of speculative tokens for speculative decoding */
  uint32_t num_speculative_tokens;

  /** Number of threads for handling input requests */
  uint32_t num_request_handling_threads;

  /** Expert parallel degree for MoE model */
  uint32_t expert_parallel_degree;

  /** Index ID for internal server ID (unique for multiple models/versions) */
  uint32_t server_idx;

  /** Maximum GPU memory utilization ratio for model inference */
  float max_memory_utilization;

  /** Model task type (generate/embed) */
  char task[XLLM_META_STRING_FIELD_MAX_LEN];

  /** NPU communication backend (lccl/hccl). Use hccl when dp is enabled */
  char communication_backend[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Instance role (DEFAULT/PREFILL/DECODE/MIX) */
  char instance_role[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Device IP address for NPU communication */
  char device_ip[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Master address for multi-node distributed serving (e.g. 10.18.1.1:9999) */
  char master_node_addr[XLLM_META_STRING_FIELD_MAX_LEN];

  /** XService server address (empty string = disable XService) */
  char xservice_addr[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Unique instance name for identification */
  char instance_name[XLLM_META_STRING_FIELD_MAX_LEN];

  /** KV cache transfer mode (PUSH/PULL) */
  char kv_cache_transfer_mode[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Log directory path (empty string = disable logging) */
  char log_dir[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Draft hf model path (empty string = no draft model) */
  char draft_model[XLLM_META_STRING_FIELD_MAX_LEN];

  /**
   * Devices to run the draft model on (e.g. npu:0, npu:0,npu:1).
   * Empty string = use the same devices as main model
   */
  char draft_devices[XLLM_META_STRING_FIELD_MAX_LEN];
} XLLM_InitLLMOptions;

/**
 * @brief Chat message structure (for ChatCompletions)
 */
typedef struct XLLM_CAPI_EXPORT XLLM_ChatMessage {
  /** Message role (system/user/assistant) */
  char role[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Message content (NULL for function call messages) */
  char* content;
} XLLM_ChatMessage;

/**
 * @brief Inference request parameters
 * @note All numeric fields are fixed-width integers with value ranges defined
 * in macros;
 */
typedef struct XLLM_CAPI_EXPORT XLLM_RequestParams {
  /** Whether to include original prompt in response */
  bool echo;

  /** Whether it's an offline request */
  bool offline;

  /** Whether to return token log probabilities */
  bool logprobs;

  /** Whether to ignore EOS token */
  bool ignore_eos;

  /** Number of completions to return per prompt */
  uint32_t n;

  /** Maximum number of tokens to generate. Must be <= model context length */
  uint32_t max_tokens;

  /** Number of sequences to generate per prompt for top-n selection */
  uint32_t best_of;

  /** SLO timeout in milliseconds (0 = unlimited) */
  uint32_t slo_ms;

  /** Beam search width (0 = disable beam search) */
  uint32_t beam_width;

  /** Number of top log probabilities to return */
  int64_t top_logprobs;

  /** Top-K sampling cutoff (-1 = 0xFFFFFFFF means disabled) */
  int64_t top_k;

  /** Top-P sampling cutoff (range: [0.0, 1.0]) */
  float top_p;

  /** Frequency penalty (range: [0.0, 2.0]) */
  float frequency_penalty;

  /** Presence penalty (range: [-2.0, 2.0]) */
  float presence_penalty;

  /** Repetition penalty. >1.0 encourages new tokens, <1.0 encourages repetition
   */
  float repetition_penalty;

  /** Sampling temperature (range: [0.0, 2.0]) */
  float temperature;

  /** Request id */
  char request_id[XLLM_META_STRING_FIELD_MAX_LEN];
} XLLM_RequestParams;

/**
 * @brief API response status codes
 */
typedef enum XLLM_CAPI_EXPORT XLLM_StatusCode {
  /** Request succeeded */
  kSuccess = 0,

  /** LLM instance not initialized */
  kNotInitialized = 1,

  /** Specified model ID not loaded */
  kModelNotFound = 2,

  /** Request timed out */
  kTimeout = 3,

  /** Invalid input parameters */
  kInvalidRequest = 4,

  /** Internal system error */
  kInternalError = 5
} XLLM_StatusCode;

/**
 * @brief Token usage statistics for inference request
 */
typedef struct XLLM_CAPI_EXPORT XLLM_Usage {
  /** Number of tokens in the prompt */
  uint32_t prompt_tokens;

  /** Number of tokens in the generated completion */
  uint32_t completion_tokens;

  /** Total tokens used (prompt + completion) */
  uint32_t total_tokens;
} XLLM_Usage;

/**
 * @brief Token log probability structure
 */
typedef struct XLLM_CAPI_EXPORT XLLM_LogProb {
  /** Token ID */
  uint32_t token_id;

  /** Log probability of the token */
  float logprob;
} XLLM_LogProb;

/**
 * @brief List of token log probabilities
 */
typedef struct XLLM_CAPI_EXPORT XLLM_LogProbs {
  /** Pointer to array of log probability entries */
  XLLM_LogProb* entries;

  /** Number of entries in the logprobs array */
  size_t entries_size;
} XLLM_LogProbs;

/**
 * @brief Inference result candidate
 */
typedef struct XLLM_CAPI_EXPORT XLLM_Choice {
  /** Index of the generated completion candidate */
  uint32_t index;

  /** Generated text for completions inference (NULL for Chat mode) */
  char* text;

  /** Generated message for chatcompletions inference (NULL for Completion mode)
   */
  XLLM_ChatMessage* message;

  /** Token log probabilities */
  XLLM_LogProbs logprobs;

  /** Reason generation stopped (stop/length/function_call) */
  char finish_reason[XLLM_META_STRING_FIELD_MAX_LEN];
} XLLM_Choice;

/**
 * @brief List of inference result candidates
 */
typedef struct XLLM_CAPI_EXPORT XLLM_Choices {
  /** Pointer to array of completion choice entries */
  XLLM_Choice* entries;

  /** Number of entries in the choices array */
  size_t entries_size;
} XLLM_Choices;

#define XLLM_ERROR_INFO_MAX_LEN 512

/**
 * @brief Inference response structure
 */
typedef struct XLLM_CAPI_EXPORT XLLM_Response {
  /** Response status code (0 = success, non-zero = error) */
  XLLM_StatusCode status_code;

  /** Error details (NULL = no error) */
  char error_info[XLLM_ERROR_INFO_MAX_LEN];

  /** Unique ID for the completion request (fixed-length string) */
  char id[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Object type (fixed to "text_completion") */
  char object[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Unix timestamp (seconds) of when the completion was created */
  int64_t created;

  /** Model name used for the completion */
  char model[XLLM_META_STRING_FIELD_MAX_LEN];

  /** List of generated completion candidates */
  XLLM_Choices choices;

  /** Token usage statistics for the request */
  XLLM_Usage usage;
} XLLM_Response;

#ifdef __cplusplus
}
#endif

#endif  // XLLM_C_TYPES_H