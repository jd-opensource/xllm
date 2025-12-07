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

#include "mm_handler.h"

#include <butil/base64.h>
#include <glog/logging.h>

#include "core/util/http_downloader.h"
#include "mm_codec.h"
#include "mm_input.h"

namespace xllm {

bool MMHandlerBase::process(const MMContent& content, MMInputItem& input) {
  if (!this->load(content, input)) {
    LOG(ERROR) << " load mm data failed";
    return false;
  }

  if (!this->decode(input)) {
    LOG(ERROR) << " decode mm data failed";
    return false;
  }

  return true;
}

bool MMHandlerBase::load_from_dataurl(const std::string& url,
                                      std::string& data) {
  size_t pos = url.find_first_of(',');
  if (pos == std::string::npos) return false;

  butil::StringPiece sub(url, pos + 1);
  return butil::Base64Decode(sub, &data);
}

bool MMHandlerBase::load_from_local(const std::string& url, std::string& data) {
  return false;
}

bool MMHandlerBase::load_from_http(const std::string& url, std::string& data) {
  BRpcDownloader helper_;
  return helper_.fetch_data(url, data);
}

bool ImageHandler::load(const MMContent& content, MMInputItem& input) {
  input.clear();

  const auto& image_url = content.image_url;
  const auto& url = image_url.url;

  if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
      0) {  // data url

    input.type_ = MMType::IMAGE;
    return this->load_from_dataurl(url, input.raw_data_);
  } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
             0) {  // http url

    input.type_ = MMType::IMAGE;
    return this->load_from_http(url, input.raw_data_);
  } else {
    LOG(ERROR) << " image url is invalid, url is " << url;
    return false;
  }
}

bool ImageHandler::decode(MMInputItem& input) {
  OpenCVImageDecoder decoder;
  return decoder.decode(input.raw_data_, input.decode_data_);
}

bool VideoHandler::load(const MMContent& content, MMInputItem& input) {
  input.clear();

  const auto& video_url = content.video_url;
  const auto& url = video_url.url;

  if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
      0) {  // data url

    input.type_ = MMType::VIDEO;
    return this->load_from_dataurl(url, input.raw_data_);
  } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
             0) {  // http url

    input.type_ = MMType::VIDEO;
    return this->load_from_http(url, input.raw_data_);
  } else {
    LOG(ERROR) << " video url is invalid, url is " << url;
    return false;
  }
}

bool VideoHandler::decode(MMInputItem& input) {
  OpenCVVideoDecoder decoder;
  return decoder.decode(input.raw_data_, input.decode_data_, input.video_meta_);
}

MMHandlerSet::MMHandlerSet() {
  handlers_["image_url"] = std::make_unique<ImageHandler>();
  handlers_["video_url"] = std::make_unique<VideoHandler>();
  // handlers_["audio_url"] = std::make_unique<AudioHandler>();
}

MMHandlerSet::~MMHandlerSet() {}

bool MMHandlerSet::process(const std::string& type,
                           const MMContent& content,
                           MMInputItem& input) {
  auto itor = handlers_.find(type);
  if (itor == handlers_.end()) {
    return false;
  }

  auto& handler = itor->second;
  return handler->process(content, input);
}

}  // namespace xllm
