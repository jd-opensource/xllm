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
#include <folly/Function.h>

#include <thread>

#include "concurrent_queue.h"
#include "util/blocking_counter.h"
namespace xllm {

class ThreadPool final {
 public:
  // a runnable is an object intended to be executed by the threadpool
  // it must be invokable with no arguments and return void.
  using Runnable = folly::Function<void()>;

  // constructors
  ThreadPool() : ThreadPool(1) {}
  // destructor
  ~ThreadPool();

  // disable copy/move constructor and assignment
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  explicit ThreadPool(size_t num_threads);
  explicit ThreadPool(size_t num_threads, Runnable init_func);

  // schedule a runnable to be executed
  int32_t schedule(Runnable runnable);

  void schedule_with_tid(Runnable runnable, size_t tid);

  bool empty() {
    return std::all_of(queues_.begin(), queues_.end(), [](auto& queue) {
      return queue.empty();
    });
  }

  size_t size() { return threads_.size(); }

 private:
  void internal_loop(size_t tid,
                     Runnable* init_func,
                     BlockingCounter* block_counter);

  std::vector<std::thread> threads_;
  std::vector<ConcurrentQueue<Runnable>> queues_;

  std::atomic<size_t> index_{0};
};

}  // namespace xllm
