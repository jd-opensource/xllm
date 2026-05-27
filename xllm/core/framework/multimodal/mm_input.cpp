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

MMErrCode MMInputTransfer::trans(const std::vector<Message>& messages,
                                 MMInput& inputs) {
  inputs.clear();
  std::vector<MMInputItem> ins;

  for (int idx = 0; idx < messages.size(); ++idx) {
    const auto& message = messages[idx];
    const auto& mmc = std::get<MMContentVec>(message.content);

    MMErrCode code = this->trans(mmc, ins, inputs.payload());
    if (code != MMErrCode::SUCCESS) {
      return code;
    }

    inputs.insert(ins);
  }
  return MMErrCode::SUCCESS;
}

MMErrCode MMInputTransfer::trans(const MMContentVec& mmc,
                                 std::vector<MMInputItem>& inputs,
                                 MMPayload& payload) {
  inputs.clear();
  for (int idx = 0; idx < mmc.size(); ++idx) {
    const auto& item = mmc[idx];
    const auto& type = item.type;

    if (type != "text") {
      MMInputItem input;
      MMErrCode code = mm_handlers_->process(type, item, input, payload);
      if (code != MMErrCode::SUCCESS) {
        return code;
      }

      inputs.emplace_back(std::move(input));
    }
  }

  return MMErrCode::SUCCESS;
}

}  // namespace xllm
