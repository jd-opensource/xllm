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

#include "threadpool.h"

#include <glog/logging.h>
#include <pthread.h>
#include <sched.h>

#include <cerrno>
#include <cstring>
#include <thread>

namespace xllm {

namespace {

int32_t bind_thread_to_cpu_core(int32_t cpu_core) {
  if (cpu_core < 0 || cpu_core >= CPU_SETSIZE) {
    LOG(ERROR) << "Invalid CPU core " << cpu_core << ", valid range is [0, "
               << CPU_SETSIZE - 1 << "]";
    return -1;
  }

  cpu_set_t current_affinity;
  CPU_ZERO(&current_affinity);
  if (sched_getaffinity(0, sizeof(cpu_set_t), &current_affinity) != 0) {
    LOG(ERROR) << "Failed to get current process affinity: " << strerror(errno);
    return -1;
  }
  if (!CPU_ISSET(cpu_core, &current_affinity)) {
    LOG(ERROR) << "CPU core " << cpu_core
               << " is not in the current process affinity set";
    return -1;
  }

  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(cpu_core, &cpu_set);

  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set) !=
      0) {
    LOG(ERROR) << "Failed to bind thread to CPU core " << cpu_core << ": "
               << strerror(errno);
    return -1;
  }

  LOG(INFO) << "Successfully bound thread to CPU core " << cpu_core;
  return 0;
}

}  // namespace

ThreadPool::ThreadPool(size_t num_threads) : ThreadPool(num_threads, nullptr) {}

ThreadPool::ThreadPool(size_t num_threads, Runnable init_func)
    : ThreadPool(num_threads, std::move(init_func), {}) {}

ThreadPool::ThreadPool(size_t num_threads, std::vector<int32_t> cpu_cores)
    : ThreadPool(num_threads, nullptr, std::move(cpu_cores)) {}

ThreadPool::ThreadPool(size_t num_threads,
                       Runnable init_func,
                       std::vector<int32_t> cpu_cores)
    : queues_(num_threads) {
  if (!cpu_cores.empty() && cpu_cores.size() != num_threads) {
    LOG(WARNING) << "ThreadPool: cpu_cores.size() (" << cpu_cores.size()
                 << ") != num_threads (" << num_threads
                 << "), CPU core binding will be skipped";
    cpu_cores.clear();
  }
  BlockingCounter counter(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    int32_t cpu_core = cpu_cores.empty() ? -1 : cpu_cores[i];
    threads_.emplace_back([this,
                           i,
                           cpu_core,
                           init_func_ptr = &init_func,
                           counter_ptr = &counter]() mutable {
      internal_loop(i, init_func_ptr, counter_ptr, cpu_core);
    });
  }
  counter.wait();
}

ThreadPool::~ThreadPool() {
  // push nullptr to the queue to signal threads to exit
  for (size_t i = 0; i < threads_.size(); ++i) {
    queues_[i].push(nullptr);
  }
  // wait for all threads to finish
  for (auto& thread : threads_) {
    thread.join();
  }
}

// schedule a runnable to be executed
int32_t ThreadPool::schedule(Runnable runnable) {
  if (runnable == nullptr) {
    return -1;
  }

  size_t current;
  size_t next;
  do {
    current = index_.load(std::memory_order_relaxed);
    next = (current + 1) % queues_.size();
  } while (!index_.compare_exchange_weak(
      current, next, std::memory_order_relaxed, std::memory_order_relaxed));
  queues_[current].push(std::move(runnable));
  return current;
}

void ThreadPool::schedule_with_tid(Runnable runnable, size_t tid) {
  if (runnable == nullptr) {
    return;
  }

  queues_[tid].push(std::move(runnable));
}

void ThreadPool::internal_loop(size_t index,
                               Runnable* init_func,
                               BlockingCounter* block_counter,
                               int32_t cpu_core) {
  if (cpu_core >= 0 && bind_thread_to_cpu_core(cpu_core) != 0) {
    LOG(WARNING) << "Thread " << index << " CPU binding to core " << cpu_core
                 << " failed, running unbound";
  }
  if (init_func != nullptr && *init_func != nullptr) {
    (*init_func)();
  }
  block_counter->decrement_count();

  while (true) {
    Runnable runnable = queues_[index].pop();

    if (runnable == nullptr) {
      // nullptr is a signal to exit
      break;
    }
    runnable();
  }
}

}  // namespace xllm
