#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "chat.pb.h"
#include "completion.pb.h"
#include "core/common/macros.h"
#include "embedding.pb.h"
#include "multimodal.pb.h"
#include "request_output.h"

namespace xllm {

struct RequestParams {
  RequestParams() = default;
  RequestParams(const proto::CompletionRequest& request,
                const std::string& x_rid,
                const std::string& x_rtime);
  RequestParams(const proto::ChatRequest& request,
                const std::string& x_rid,
                const std::string& x_rtime);
  RequestParams(const proto::MMChatRequest& request,
                const std::string& x_rid,
                const std::string& x_rtime);
  RequestParams(const proto::EmbeddingRequest& request,
                const std::string& x_rid,
                const std::string& x_rtime);

  bool verify_params(OutputCallback callback) const;

  // request id
  std::string request_id;
  std::string service_request_id = "";
  std::string x_request_id;
  std::string x_request_time;

  bool streaming = false;

  // number of tokens to generate. truncted to model's max context length.
  uint32_t max_tokens = 5120;

  // number of sequences to generate for each prompt.
  uint32_t n = 1;

  // number of sequences to generate for each prompt and select n best among.
  std::optional<uint32_t> best_of;

  // whether to include the original prompt in the completion response.
  bool echo = false;

  // frequency penalty to reduce the likelihood of generating the same word
  // multiple times. values between [0.0, 2.0]. 0.0 means no penalty. default =
  // 0.0 Positive values penalize new tokens based on their existing frequency
  // in the text.
  float frequency_penalty = 0.0;

  // presence penalty to reduce the likelihood of generating words already in
  // the prompt. values between [-2.0, 2.0]. Positive values penalize new tokens
  // based on their existing in the prompt. default = 0.0
  float presence_penalty = 0.0;

  // repetition penalty to penalize new tokens based on their occurence in the
  // text. values > 1.0 encourage the model to use new tokens, while values
  // < 1.0 encourage the model to repeat tokens. default = 1.0
  float repetition_penalty = 1.0;

  // temperature of the sampling, between [0, 2]. default = 0.0
  // higher value will make the ouput more random.
  float temperature = 0.0;

  // top_p sampling cutoff, between [0.0, 1.0]. default = 1.0
  float top_p = 1.0;

  // top_k sampling cutoff. default = -1 to disable.
  int64_t top_k = -1;

  // whether to return the log probabilities of the tokens. default = false.
  bool logprobs = false;

  // number of top log probabilities to return. default = 0.
  int64_t top_logprobs = 0;

  // whether to skip special tokens in the output text. default = true.
  bool skip_special_tokens = true;

  // whether to ignore the end of sequence token. default = false.
  bool ignore_eos = false;

  // wheteher to get the embeddings of the tokens. used by embeddings model.
  bool is_embeddings = false;

  // the list of strings to stop generating further tokens.
  // the output will contain the stop string.
  std::optional<std::vector<std::string>> stop;

  // the list of token ids to stop generating further tokens.
  std::optional<std::vector<int32_t>> stop_token_ids;

  // decode address.
  std::string decode_address;

  std::vector<Tool> tools;
  std::string tool_choice = "auto";
  bool has_tools() const { return !tools.empty(); }
};

}  // namespace xllm
