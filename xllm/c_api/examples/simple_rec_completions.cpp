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

#include <unistd.h>

#include <cstring>
#include <iostream>
#include <random>

#include "rec.h"

std::string devices = "cuda:1";
std::string model_name = "Qwen3-8B";
std::string model_path = "/export/home/models/Qwen3-8B";
#define DEFAULT_TOKEN_SIZE 100

XLLM_REC_Handler* service_startup_hook() {
  XLLM_REC_Handler* rec_handler = xllm_rec_create();

  // If there is no separate setting, init_options can be passed as nullptr, and
  // the default value(XLLM_INIT_REC_OPTIONS_DEFAULT) will be used
  XLLM_InitOptions init_options;
  xllm_rec_init_options_default(&init_options);
  snprintf(
      init_options.log_dir, sizeof(init_options.log_dir), "/export/xllm/log");

  bool ret = xllm_rec_initialize(
      rec_handler, model_path.c_str(), devices.c_str(), &init_options);
  if (!ret) {
    std::cout << "REC init failed" << std::endl;
    xllm_rec_destroy(rec_handler);
    return nullptr;
  }

  std::cout << "REC init successfully" << std::endl;

  return rec_handler;
}

void service_stop_hook(XLLM_REC_Handler* rec_handler) {
  xllm_rec_destroy(rec_handler);
  std::cout << "REC stop" << std::endl;
}

int generate_random_int(int min, int max) {
  if (min > max) {
    throw std::invalid_argument("min cannot be greater than max");
  }

  static std::random_device rd;
  static std::mt19937 gen(rd());

  std::uniform_int_distribution<int> dist(min, max);

  return dist(gen);
}

int main(int argc, char** argv) {
  XLLM_REC_Handler* rec_handler = service_startup_hook();
  if (nullptr == rec_handler) {
    return -1;
  }

  // If there is no separate setting, request_params can be passed as nullptr,
  // and the default value(XLLM_REQUEST_PARAMS_DEFAULT) will be used
  XLLM_RequestParams request_params;
  xllm_rec_request_params_default(&request_params);
  request_params.max_tokens = 100;

  size_t token_size = DEFAULT_TOKEN_SIZE;
  int token_ids[DEFAULT_TOKEN_SIZE] = {0};
  for (int i = 0; i < token_size; i++) {
    token_ids[i] = generate_random_int(100, 150000);
  }

  XLLM_Response* resp = xllm_rec_token_completions(rec_handler,
                                                   model_name.c_str(),
                                                   token_ids,
                                                   token_size,
                                                   10000,
                                                   &request_params);
  if (nullptr == resp) {
    std::cout << "REC completions failed, response is nullptr" << std::endl;
    service_stop_hook(rec_handler);
    return -1;
  }

  if (resp->status_code != XLLM_StatusCode::kSuccess) {
    std::cout << "REC completions failed, status code:" << resp->status_code
              << ", error info:" << resp->error_info << std::endl;
  } else {
    std::cout << "REC completions successfully" << std::endl;

    if (nullptr != resp->choices.entries) {
      for (int i = 0; i < resp->choices.entries_size; ++i) {
        XLLM_Choice& choice = resp->choices.entries[i];
        for (int j = 0; j < choice.logprobs.entries_size; j++) {
          XLLM_LogProb& logprob = choice.logprobs.entries[j];
          std::cout << "xllm answer[" << choice.index
                    << "]: token id=" << logprob.token_id
                    << ", token logprob=" << logprob.logprob << std::endl;
        }
      }
    }
  }

  xllm_rec_free_response(resp);

  service_stop_hook(rec_handler);

  return 0;
}