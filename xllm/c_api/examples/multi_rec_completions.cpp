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

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rec.h"
#include "types.h"

/**
 * 双 NPU chip：npu:0 / npu:1 各一个 REC 实例。两个 handler 使用相同
 * `XLLM_InitOptions`，只通过 `devices` 参数选择目标 NPU。OneRec 需
 * `sparse_embedding`，hidden 同 config（本例 2048）。
 *
 * 并发说明：
 * - 默认模式创建两个 REC handler，并使用两个 host 线程同时调用
 *   `xllm_rec_input_tensors_completions`。
 * - `--interleaved` 保留单 host 线程交错请求模式，用于和并发模式对照。
 */

namespace {

// xllm.proto.DataType: FLOAT = 1
constexpr int32_t kProtoDataTypeFloat = 1;

const char* kModelPath =
    "/export/home/liuhan37/xllm-so/xllm/model_repository/"
    "content-onerec/2026031717090608/tf_models/graph2";
const char* kModelId = "graph2";

// 与 `graph2/config.json` 中 d_model / decoder 配置一致
const int64_t kOneRecHiddenSize = 2048;
const int64_t kDemoSparseSeqLen = 8;

const uint32_t kInferTimeoutMs = 100000U;

// 无 argv 时默认并发推多长时间（秒）
const int32_t kDefaultConcurrentSeconds = 10;

// 在长时间循环中每隔多少次成功推理打印一次，避免刷日志
const int32_t kProgressLogInterval = 50;

std::mutex g_io_mutex;

std::string device_for_chip(int32_t chip_index) {
  if (chip_index == 0) {
    return "npu:0";
  }
  return "npu:1";
}

bool init_rec_handler_for_chip(int32_t chip_index,
                               XLLM_REC_Handler* handler,
                               XLLM_InitOptions* init_options) {
  xllm_rec_init_options_default(init_options);
  const char* kLogDir = "./Logs";
  std::strncpy(
      init_options->log_dir, kLogDir, XLLM_META_STRING_FIELD_MAX_LEN - 1);
  init_options->log_dir[XLLM_META_STRING_FIELD_MAX_LEN - 1] = '\0';

  init_options->max_memory_utilization = 0.9F;

  std::string devices = device_for_chip(chip_index);
  return xllm_rec_initialize(
      handler, kModelPath, devices.c_str(), init_options);
}

bool run_input_tensors_completions(XLLM_REC_Handler* handler,
                                   const char* log_tag,
                                   int32_t request_seq,
                                   bool log_detail_on_success) {
  const int64_t numel = kDemoSparseSeqLen * kOneRecHiddenSize;
  std::vector<float> sparse_data(static_cast<size_t>(numel), 0.0F);
  for (int64_t i = 0; i < numel; ++i) {
    const float v = 0.001F * static_cast<float>((i + request_seq) % 7);
    sparse_data[static_cast<size_t>(i)] = v;
  }

  std::array<int64_t, 2> shape = {kDemoSparseSeqLen, kOneRecHiddenSize};
  XLLM_InferInputTensorDesc tensor_desc;
  std::memset(&tensor_desc, 0, sizeof(tensor_desc));
  tensor_desc.name = "sparse_embedding";
  tensor_desc.data_type = kProtoDataTypeFloat;
  tensor_desc.shape = shape.data();
  tensor_desc.shape_len = 2U;
  tensor_desc.data = sparse_data.data();
  tensor_desc.num_elements = static_cast<size_t>(numel);

  XLLM_RequestParams request_params;
  xllm_rec_request_params_default(&request_params);
  request_params.max_tokens = 3U;
  request_params.beam_width = 128U;
  request_params.logprobs = true;
  request_params.top_k = 128;
  request_params.top_logprobs = 128;

  XLLM_InferInputTensorDesc tensors[1] = {tensor_desc};

  XLLM_Response* resp = xllm_rec_input_tensors_completions(
      handler, kModelId, tensors, 1U, kInferTimeoutMs, &request_params);
  if (resp == nullptr) {
    std::lock_guard<std::mutex> lock(g_io_mutex);
    std::cout << log_tag << " req=" << request_seq
              << " xllm_rec_input_tensors_completions returned null"
              << std::endl;
    return false;
  }

  if (resp->status_code != XLLM_StatusCode::kSuccess) {
    std::lock_guard<std::mutex> lock(g_io_mutex);
    std::cout << log_tag << " req=" << request_seq
              << " failed, status=" << static_cast<int32_t>(resp->status_code)
              << " info="
              << (resp->error_info != nullptr ? resp->error_info : "")
              << std::endl;
    xllm_rec_free_response(resp);
    return false;
  }

  if (log_detail_on_success) {
    std::lock_guard<std::mutex> lock(g_io_mutex);
    std::cout << log_tag << " ok, choices=" << resp->choices.entries_size
              << std::endl;
    if (resp->choices.entries != nullptr) {
      for (size_t i = 0; i < resp->choices.entries_size; ++i) {
        XLLM_Choice& choice = resp->choices.entries[i];
        std::cout << log_tag << " choice[" << choice.index
                  << "] token_size=" << choice.token_size << std::endl;
      }
    }
  }

  xllm_rec_free_response(resp);
  return true;
}

void concurrent_infer_loop(XLLM_REC_Handler* handler,
                           const char* log_tag,
                           int32_t run_seconds,
                           std::atomic<int32_t>* any_fail,
                           int32_t* out_total_ok) {
  const auto t_end =
      std::chrono::steady_clock::now() + std::chrono::seconds(run_seconds);
  int32_t seq = 0;
  int32_t ok = 0;
  while (std::chrono::steady_clock::now() < t_end) {
    if (any_fail->load(std::memory_order_acquire) != 0) {
      *out_total_ok = ok;
      return;
    }
    ++seq;
    if (!run_input_tensors_completions(handler, log_tag, seq, false)) {
      any_fail->store(1, std::memory_order_release);
      *out_total_ok = ok;
      return;
    }
    ++ok;
    if (ok % kProgressLogInterval == 0) {
      std::lock_guard<std::mutex> lock(g_io_mutex);
      std::cout << log_tag << " progress: " << ok << " ok requests"
                << std::endl;
    }
  }
  *out_total_ok = ok;
  std::lock_guard<std::mutex> lock(g_io_mutex);
  std::cout << log_tag << " done: " << ok << " total ok in " << run_seconds
            << "s window" << std::endl;
}

// 单线程内交替向两个 handler 发推理，在时间上重叠负载且避免多线程同时进入 C
// API。
bool interleaved_infer_two_handlers(XLLM_REC_Handler* h0,
                                    XLLM_REC_Handler* h1,
                                    int32_t run_seconds,
                                    int32_t* out_c0,
                                    int32_t* out_c1) {
  const auto t_end =
      std::chrono::steady_clock::now() + std::chrono::seconds(run_seconds);
  int32_t seq0 = 0;
  int32_t seq1 = 0;
  int32_t ok0 = 0;
  int32_t ok1 = 0;
  while (std::chrono::steady_clock::now() < t_end) {
    if (!run_input_tensors_completions(h0, "[chip0]", ++seq0, false)) {
      *out_c0 = ok0;
      *out_c1 = ok1;
      return false;
    }
    ++ok0;
    if (!run_input_tensors_completions(h1, "[chip1]", ++seq1, false)) {
      *out_c0 = ok0;
      *out_c1 = ok1;
      return false;
    }
    ++ok1;
    const int32_t total_rounds = ok0;
    if (total_rounds % kProgressLogInterval == 0) {
      std::lock_guard<std::mutex> lock(g_io_mutex);
      std::cout << "[interleaved] rounds=" << total_rounds << " (chip0+chip1)"
                << std::endl;
    }
  }
  *out_c0 = ok0;
  *out_c1 = ok1;
  std::lock_guard<std::mutex> lock(g_io_mutex);
  std::cout << "[interleaved] done: chip0 ok=" << ok0 << " chip1 ok=" << ok1
            << " in " << run_seconds << "s" << std::endl;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int32_t run_seconds = kDefaultConcurrentSeconds;
  bool use_parallel_threads = true;
  for (int32_t i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--threads") == 0) {
      use_parallel_threads = true;
      continue;
    }
    if (std::strcmp(argv[i], "--interleaved") == 0) {
      use_parallel_threads = false;
      continue;
    }
    if (run_seconds == kDefaultConcurrentSeconds) {
      const int32_t parsed = static_cast<int32_t>(std::atoi(argv[i]));
      if (parsed > 0) {
        run_seconds = parsed;
      }
    }
  }

  std::cout << "Mode: "
            << (use_parallel_threads ? "parallel threads"
                                     : "interleaved (single client thread)")
            << std::endl;

  std::cout << "Init REC instance 1 on npu:0 (chip0)." << std::endl;
  XLLM_REC_Handler* rec_handler_01 = xllm_rec_create();
  if (rec_handler_01 == nullptr) {
    std::cout << "xllm_rec_create (1) failed" << std::endl;
    return -1;
  }
  XLLM_InitOptions init_options_01;
  if (!init_rec_handler_for_chip(0, rec_handler_01, &init_options_01)) {
    std::cout << "xllm_rec_initialize (1) failed" << std::endl;
    xllm_rec_destroy(rec_handler_01);
    return -1;
  }
  std::cout << "REC instance 1 init ok." << std::endl;

  std::cout << "Init REC instance 2 on npu:1 (chip1)." << std::endl;
  XLLM_REC_Handler* rec_handler_02 = xllm_rec_create();
  if (rec_handler_02 == nullptr) {
    std::cout << "xllm_rec_create (2) failed" << std::endl;
    xllm_rec_destroy(rec_handler_01);
    return -1;
  }
  XLLM_InitOptions init_options_02;
  if (!init_rec_handler_for_chip(1, rec_handler_02, &init_options_02)) {
    std::cout << "xllm_rec_initialize (2) failed" << std::endl;
    xllm_rec_destroy(rec_handler_02);
    xllm_rec_destroy(rec_handler_01);
    return -1;
  }
  std::cout << "REC instance 2 init ok." << std::endl;

  std::cout << "Run window " << run_seconds
            << " s (e.g. ./multi_rec_completions 30 [--interleaved])"
            << std::endl;

  int32_t count0 = 0;
  int32_t count1 = 0;
  int32_t ret = 0;
  if (use_parallel_threads) {
    std::atomic<int32_t> any_fail{0};
    std::thread t0(concurrent_infer_loop,
                   rec_handler_01,
                   "[chip0]",
                   run_seconds,
                   &any_fail,
                   &count0);
    std::thread t1(concurrent_infer_loop,
                   rec_handler_02,
                   "[chip1]",
                   run_seconds,
                   &any_fail,
                   &count1);
    t0.join();
    t1.join();
    if (any_fail.load(std::memory_order_acquire) != 0) {
      ret = -1;
    }
    if (count0 == 0 || count1 == 0) {
      std::cout << "Concurrent verification failed: both handlers must finish "
                   "at least one request"
                << std::endl;
      ret = -1;
    }
  } else {
    if (!interleaved_infer_two_handlers(
            rec_handler_01, rec_handler_02, run_seconds, &count0, &count1)) {
      ret = -1;
    }
  }

  xllm_rec_destroy(rec_handler_02);
  xllm_rec_destroy(rec_handler_01);

  if (ret != 0) {
    std::cout << "Exiting with failure (see errors above)." << std::endl;
    return -1;
  }
  std::cout << "Done. chip0 ok=" << count0 << " chip1 ok=" << count1
            << std::endl;
  return 0;
}