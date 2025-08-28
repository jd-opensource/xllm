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

#include "blocking_counter.h"

#include <gtest/gtest.h>

#include "util/threadpool.h"

namespace xllm {

TEST(BlockingCounterTest, BasicTest) {
  BlockingCounter counter(1);
  counter.decrement_count();
  counter.wait();
  EXPECT_TRUE(true);
}

TEST(BlockingCounterTest, TwoThreadTest) {
  ThreadPool threadpool(1);
  BlockingCounter counter(2);

  int called = 0;
  threadpool.schedule([&counter, &called]() {
    counter.decrement_count();
    ++called;
  });
  counter.decrement_count();
  ++called;
  counter.wait();
  EXPECT_EQ(2, called);
}

TEST(BlockingCounterTest, MultiThreadTest) {
  ThreadPool threadpool(4);
  BlockingCounter counter(5);

  int called = 0;
  threadpool.schedule([&counter, &called]() {
    counter.decrement_count();
    ++called;
  });
  threadpool.schedule([&counter, &called]() {
    counter.decrement_count();
    ++called;
  });
  threadpool.schedule([&counter, &called]() {
    counter.decrement_count();
    ++called;
  });
  threadpool.schedule([&counter, &called]() {
    counter.decrement_count();
    ++called;
  });
  counter.decrement_count();
  ++called;

  counter.wait();
  EXPECT_EQ(5, called);
}

TEST(BlockingCounterTest, WaitTimeoutTest) {
  ThreadPool threadpool(2);
  BlockingCounter counter(3);

  int called = 0;
  threadpool.schedule([&counter, &called]() {
    counter.decrement_count();
    ++called;
  });

  counter.decrement_count();
  ++called;

  const std::chrono::milliseconds timeout(100);
  counter.wait_for(timeout);
  EXPECT_EQ(2, called);
}

}  // namespace xllm
