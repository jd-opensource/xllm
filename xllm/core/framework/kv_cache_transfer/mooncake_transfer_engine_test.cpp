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

#include "framework/kv_cache_transfer/mooncake_transfer_engine.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>

namespace xllm {

TEST(MooncakeTransferEngineServiceTest, OpenSessionRejectsMissingAddr) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  proto::Status response;
  brpc::Controller cntl;

  service.OpenSession(&cntl, &request, &response, nullptr);

  EXPECT_FALSE(response.ok());
}

TEST(MooncakeTransferEngineServiceTest, CloseSessionRejectsMissingAddr) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  proto::Status response;
  brpc::Controller cntl;

  service.CloseSession(&cntl, &request, &response, nullptr);

  EXPECT_FALSE(response.ok());
}

TEST(MooncakeTransferEngineServiceTest, CloseSessionWithoutHandleReturnsTrue) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  request.set_addr("127.0.0.1:5001");
  proto::Status response;
  brpc::Controller cntl;

  service.CloseSession(&cntl, &request, &response, nullptr);

  EXPECT_TRUE(response.ok());
}

}  // namespace xllm
