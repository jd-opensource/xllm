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

#include "mm_input.h"

#include <glog/logging.h>

#include "mm_handler.h"

namespace xllm {

MMInputTransfer::MMInputTransfer() {
  mm_handlers_ = std::make_unique<MMHandlerSet>();
}

MMInputTransfer::~MMInputTransfer() {}

bool MMInputTransfer::trans(const std::vector<Message>& messages,
                            MMInput& inputs) {
  inputs.clear();
  std::vector<MMInputItem> ins;

  for (int idx = 0; idx < messages.size(); ++idx) {
    const auto& message = messages[idx];
    const auto& mmc = std::get<MMContentVec>(message.content);

    if (!this->trans(mmc, ins, inputs.payload_)) {
      return false;
    }

    inputs.insert(ins);
  }
  return true;
}

bool MMInputTransfer::trans(const MMContentVec& mmc,
                            std::vector<MMInputItem>& inputs,
                            MMPayload& payload) {
  inputs.clear();
  for (int idx = 0; idx < mmc.size(); ++idx) {
    const auto& item = mmc[idx];
    const auto& type = item.type;

    if (type != "text") {
      MMInputItem input;
      if (!mm_handlers_->process(type, item, input, payload)) {
        return false;
      }

      inputs.emplace_back(std::move(input));
    }
  }

  return true;
}

}  // namespace xllm
