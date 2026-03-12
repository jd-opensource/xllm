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

std::string devices = "cuda:3";
std::string model_name = "homepage_qwen_06b_6_raw";
std::string model_path = "/export/home/models/homepage_qwen_06b_6_raw";

XLLM_REC_Handler* service_startup_hook() {
  XLLM_REC_Handler* rec_handler = xllm_rec_create();

  // If there is no separate setting, init_options can be passed as nullptr, and
  // the default value(XLLM_INIT_REC_OPTIONS_DEFAULT) will be used
  XLLM_InitOptions init_options;
  xllm_rec_init_options_default(&init_options);

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

int main(int argc, char** argv) {
  XLLM_REC_Handler* rec_handler = service_startup_hook();
  if (nullptr == rec_handler) {
    return -1;
  }

  // If there is no separate setting, request_params can be passed as nullptr,
  // and the default value(XLLM_REQUEST_PARAMS_DEFAULT) will be used
  XLLM_RequestParams request_params;
  xllm_rec_request_params_default(&request_params);
  request_params.max_tokens = 3;
  request_params.beam_width = 128;
  request_params.logprobs = true;
  // request_params.temperature = 1.0;
  request_params.top_k = 128;
  request_params.top_logprobs = 128;
  // request_params.top_p = 1.0;
  // request_params.repetition_penalty = 1.0;

  std::vector<int32_t> token_ids = {
      151644, 8948,   198,    56568,  101909, 101215, 104799, 101914, 101057,
      3837,   103929, 100032, 44956,  15946,  55338,  45943,  104570, 11622,
      105801, 72881,  64559,  307,    71817,  51463,  3837,   56568,  107618,
      100345, 20002,  104754, 72651,  105565, 45943,  116951, 101034, 67949,
      72651,  109348, 36407,  104538, 20002,  104326, 87267,  72651,  109348,
      1773,   151645, 198,    151644, 872,    198,    20002,  21,     15,
      35727,  31843,  36667,  59879,  20450,  99805,  32044,  72651,  105565,
      45943,  32044,  113507, 153479, 155828, 160439, 11,     153479, 157177,
      160439, 11,     153479, 155828, 160439, 11,     153479, 155828, 160439,
      11,     153479, 155828, 160439, 11,     153479, 155828, 160439, 11,
      155622, 158228, 160337, 11,     152907, 158228, 159858, 11,     153036,
      158228, 160333, 11,     153258, 159797, 160105, 11,     153186, 157627,
      160740, 11,     152907, 158228, 160680, 11,     154562, 157329, 160321,
      11,     153326, 157680, 163928, 11,     153258, 159634, 160105, 11,
      152847, 157129, 162841, 11,     152847, 157399, 162841, 11,     152847,
      158228, 163388, 11,     153036, 159807, 162840, 11,     154562, 157329,
      160321, 11,     154562, 156839, 160321, 11,     154562, 158181, 160321,
      11,     153326, 158534, 163886, 11,     153326, 157177, 163041, 11,
      155622, 158228, 163359, 11,     152569, 155800, 162738, 11,     153390,
      158228, 160357, 11,     152663, 157649, 162738, 11,     155193, 158667,
      162738, 11,     155622, 158228, 160706, 11,     151685, 158473, 162738,
      11,     152907, 158228, 162653, 11,     151876, 158228, 159909, 11,
      152907, 158228, 162407, 11,     152907, 158228, 163551, 11,     151685,
      158473, 162738, 11,     152686, 155927, 162029, 11,     152663, 158228,
      161841, 11,     152686, 155927, 162603, 11,     153516, 157280, 161980,
      11,     153516, 159807, 160708, 11,     153516, 157900, 163856, 11,
      153516, 155967, 161020, 11,     153516, 157280, 160838, 11,     153200,
      157591, 162582, 11,     151924, 158696, 160358, 11,     154562, 159113,
      160860, 11,     153386, 159086, 161519, 11,     154625, 159807, 160781,
      11,     153479, 155828, 160439, 11,     153479, 155828, 160439, 11,
      153479, 157177, 160439, 11,     153479, 155828, 160439, 11,     154213,
      157866, 160523, 11,     153036, 156918, 163610, 11,     153036, 157351,
      160974, 11,     153688, 158228, 160337, 11,     155507, 159807, 162736,
      11,     155370, 159219, 161059, 11,     155002, 158118, 160019, 11,
      155370, 159219, 161059, 11,     153792, 159022, 161003, 11,     155576,
      155927, 161581, 11,     155576, 155927, 163189, 11,     155576, 159630,
      162853, 11,     155576, 159630, 163527, 11,     155576, 159630, 162164,
      11,     155576, 158048, 163339, 11,     155576, 157177, 163339, 11,
      155576, 159630, 163527, 11,     155576, 157177, 163339, 11,     155576,
      157680, 163339, 11,     155576, 159630, 160653, 11,     155576, 159630,
      162153, 11,     155576, 159630, 161747, 11,     155576, 157505, 163339,
      11,     153831, 158228, 160026, 11,     153390, 158228, 161841, 11,
      153831, 156324, 162738, 11,     153390, 158228, 161491, 11,     153390,
      159145, 162738, 11,     155507, 158473, 162738, 11,     153831, 157649,
      162738, 11,     155507, 157770, 162738, 11,     153390, 158228, 161033,
      11,     155507, 158473, 162738, 11,     153390, 158228, 160824, 11,
      153479, 157649, 160439, 11,     153479, 157649, 160439, 11,     153479,
      155828, 160439, 11,     153479, 157649, 160439, 11,     153479, 157649,
      160439, 11,     153479, 157649, 160439, 11,     153849, 159380, 162841,
      11,     152663, 158107, 162738, 11,     152271, 157371, 161110, 11,
      152663, 157176, 160199, 11,     154936, 158966, 162841, 11,     153390,
      158228, 161491, 11,     153036, 158228, 162840, 11,     155646, 158228,
      162408, 11,     152663, 156814, 162738, 11,     152569, 158473, 162738,
      11,     155646, 158228, 161308, 11,     152663, 158228, 163631, 11,
      155370, 159786, 163029, 11,     153534, 159283, 161094, 11,     153534,
      157756, 163778, 11,     151905, 156698, 163573, 11,     151905, 156698,
      161534, 11,     151905, 156698, 162140, 11,     153534, 157931, 161817,
      11,     153534, 157121, 161059, 11,     154826, 158585, 163433, 11,
      154826, 158585, 160756, 11,     154826, 157666, 161504, 11,     154826,
      157351, 161808, 11,     154826, 158585, 161062, 11,     154826, 157666,
      161504, 11,     154826, 156537, 163635, 11,     155370, 159219, 161059,
      11,     155370, 156903, 160381, 11,     155370, 156903, 160381, 11,
      155370, 159219, 162223, 11,     155370, 159330, 162223, 11,     153464,
      159219, 161059, 11,     154809, 156903, 160381, 11,     153464, 156878,
      162223, 11,     154809, 157794, 162010, 11,     154809, 159219, 161059,
      11,     151893, 159807, 162666, 11,     151893, 158534, 160890, 11,
      153326, 157177, 163620, 11,     153326, 159462, 163041, 11,     152663,
      156348, 162738, 11,     152663, 158473, 162736, 11,     152463, 156537,
      160873, 11,     155507, 157176, 162738, 11,     155193, 158473, 162738,
      11,     152663, 157649, 162738, 11,     152663, 158107, 162738, 11,
      152663, 155780, 162738, 11,     152663, 158473, 162738, 11,     152663,
      157649, 162738, 11,     152663, 157649, 162738, 11,     152663, 155828,
      162738, 11,     152663, 158621, 162738, 11,     152663, 157176, 162738,
      11,     155646, 158228, 160017, 11,     155682, 158228, 162859, 67949,
      103969, 72651,  109348, 17714,  155646, 158228, 162234, 1773,   104210,
      67949,  9370,   72651,  45943,  9370,   111450, 37945,  104538, 20002,
      104326, 104309, 72651,  9370,   16,     15,     18947,  45943,  3837,
      11622,  107463, 17992,  71817,  17177,  99859,  1773,   151645, 198,
      151644, 77091,  198};

  size_t token_size = token_ids.size();
  const int32_t* token_ids_ptr = token_ids.data();

  XLLM_Response* resp = xllm_rec_token_completions(rec_handler,
                                                   model_name.c_str(),
                                                   token_ids_ptr,
                                                   token_size,
                                                   100000,
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
    std::cout << "REC completions successfully, size:"
              << resp->choices.entries_size << std::endl;

    if (nullptr != resp->choices.entries) {
      for (int i = 0; i < resp->choices.entries_size; ++i) {
        XLLM_Choice& choice = resp->choices.entries[i];
        std::cout << "token size: " << choice.token_size
                  << ",logprobs size:" << choice.logprobs.entries_size
                  << std::endl;

        for (int j = 0; j < choice.token_size; j++) {
          std::cout << "xllm answer[" << choice.index
                    << "]: token id=" << choice.token_ids[j] << std::endl;
        }

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