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

#include "mm_input_helper.h"

#include <opencv2/opencv.hpp>

#include "butil/base64.h"

namespace xllm {

class OpenCVImageDecoder {
 public:
  bool decode(const std::string& raw_data, torch::Tensor& t) {
    cv::Mat buffer(1, raw_data.size(), CV_8UC1, (void*)raw_data.data());
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (image.empty()) {
      LOG(INFO) << " opencv image decode failed";
      return false;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);  // RGB

    torch::Tensor tensor = torch::from_blob(
        image.data, {image.rows, image.cols, 3}, torch::kUInt8);

    t = tensor.permute({2, 0, 1}).clone();  // [C, H, W]
    return true;
  }
};

class Handler {
 public:
  bool process(const proto::MMInputData& msg, MMInputItem& input) {
    if (!this->load(msg, input)) {
      LOG(ERROR) << " load mm data failed";
      return false;
    }

    if (!this->decode(input)) {
      LOG(ERROR) << " decode mm data failed";
      return false;
    }

    return true;
  }

  virtual bool load(const proto::MMInputData& msg, MMInputItem& input) = 0;
  virtual bool decode(MMInputItem& input) = 0;

 protected:
  bool load_from_dataurl(const std::string& url, std::string& data) {
    size_t pos = url.find_first_of(',');
    if (pos == std::string::npos) return false;

    butil::StringPiece sub(url, pos + 1);
    return butil::Base64Decode(sub, &data);
  }

  bool load_from_local(const std::string& url, std::string& data) {
    return false;
  }

  bool load_from_http(const std::string& url, std::string& data) {
    return false;
  }
};

class ImageHandler : public Handler {
 public:
  ImageHandler() : dataurl_prefix_("data:image") {}

  virtual bool load(const proto::MMInputData& msg, MMInputItem& input) {
    input.clear();

    const auto& image_url = msg.image_url();
    const auto& url = image_url.url();

    if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
        0) {  // data url

      input.type_ = MMType::IMAGE;
      return this->load_from_dataurl(url, input.raw_data_);
    } else {
      return false;
    }
  }

  virtual bool decode(MMInputItem& input) {
    OpenCVImageDecoder decoder;
    return decoder.decode(input.raw_data_, input.decode_data_);
  }

 private:
  const std::string dataurl_prefix_;
};

class MMHandlerSet {
 public:
  MMHandlerSet() {
    handlers_["image_url"] = std::make_unique<ImageHandler>();
    // handlers_["video_url"] = std::make_unique<VideoHandler>();
    // handlers_["audio_url"] = std::make_unique<AudioHandler>();
  }

  bool process(const std::string& type,
               const proto::MMInputData& msg,
               MMInputItem& input) {
    auto itor = handlers_.find(type);
    if (itor == handlers_.end()) {
      return false;
    }

    auto& handler = itor->second;
    return handler->process(msg, input);
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<Handler> > handlers_;
};

MMInputHelper::MMInputHelper() {
  mm_handlers_ = std::make_unique<MMHandlerSet>();
}

MMInputHelper::~MMInputHelper() {}

bool MMInputHelper::trans(const MMChatMessageVec& vec,
                          std::vector<Message>& messages,
                          MMInputItemVec& inputs) {
  messages.clear();
  inputs.clear();

  messages.reserve(vec.size());
  inputs.reserve(vec.size());

  for (int idx = 0; idx < vec.size(); ++idx) {
    const auto& chat = vec[idx];
    const auto& role = chat.role();
    const auto& content = chat.content();

    Message::MMContentVec mmc;
    MMInputItemVec ins;
    if (!this->trans(content, mmc, ins)) {
      return false;
    }

    messages.emplace_back(role, mmc);
    inputs.insert(inputs.end(), ins.begin(), ins.end());
  }
  return true;
}

bool MMInputHelper::trans(const MMInputDataVec& vec,
                          Message::MMContentVec& mmc,
                          MMInputItemVec& inputs) {
  mmc.clear();
  inputs.clear();

  for (int idx = 0; idx < vec.size(); ++idx) {
    const auto& item = vec[idx];
    const auto& type = item.type();

    if (type == "text") {
      mmc.emplace_back(type, item.text());
    } else {
      MMInputItem input;
      if (!mm_handlers_->process(type, item, input)) {
        return false;
      }

      mmc.emplace_back(type);
      inputs.emplace_back(input);
    }
  }

  return true;
}

}  // namespace xllm
