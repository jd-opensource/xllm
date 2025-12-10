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

#include "core/common/message.h"
#include "core/common/types.h"
#include "multimodal.pb.h"

namespace xllm {

template <typename Call>
bool build_messages(const google::protobuf::RepeatedPtrField<
                        xllm::proto::MMChatMessage>& req_messages,
                    std::vector<Message>& out_messages,
                    std::shared_ptr<Call> call,
                    int image_limit) {
  out_messages.clear();
  out_messages.reserve(req_messages.size());

  for (const auto& req_message : req_messages) {
    MMContentVec contents;

    for (const auto& input : req_message.content()) {
      auto& item = const_cast<::xllm::proto::MMInputData&>(input);

      if (item.type() == "text") {
        contents.emplace_back(item.type(), *item.release_text());

      } else if (item.type() == "image_url") {
        ImageURL image_url;
        image_url.url = std::move(*item.mutable_image_url()->release_url());
        contents.emplace_back(item.type(), image_url);

      } else if (item.type() == "video_url") {
        VideoURL video_url;
        video_url.url = std::move(*item.mutable_video_url()->release_url());
        contents.emplace_back(item.type(), video_url);

      } else if (item.type() == "audio_url") {
        AudioURL audio_url;
        audio_url.url = std::move(*item.mutable_audio_url()->release_url());
        contents.emplace_back(item.type(), audio_url);
      } else if (item.type() == "image_embedding") {
        contents.emplace_back("image_embedding", item.image_embedding());
      } else if (item.type() == "video_embedding") {
        contents.emplace_back("video_embedding", item.video_embedding());
      } else if (item.type() == "audio_embedding") {
        contents.emplace_back("audio_embedding", item.audio_embedding());
      } else {
        call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                                "message content type is invalid.");
        return false;
      }
    }

    out_messages.emplace_back(req_message.role(), std::move(contents));
  }

  for (auto& msg : out_messages) {
    if (msg.calc_count("image_url") > image_limit) {
      call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                              "Number of images in a single message exceeds "
                              "the allowed image limit.");
      return false;
    }
  }

  return true;
};

}  // namespace xllm