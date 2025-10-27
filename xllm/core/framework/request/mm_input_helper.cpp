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

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <bthread/rwlock.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>

#include "butil/base64.h"
#include "mm_codec.h"

namespace xllm {

class FileDownloadHelper {
 public:
  FileDownloadHelper() {}
  ~FileDownloadHelper() {}
  std::string parse_url(const std::string& url) {
    size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
      LOG(ERROR)
          << "Error: Invalid URL, missing protocol (http:// or https://)";
    }
    size_t host_start = scheme_end + 3;
    size_t path_pos = url.find('/', host_start);
    if (path_pos == std::string::npos) {
      LOG(ERROR) << "Error: No path in URL";
    }
    return url.substr(host_start, path_pos - host_start);
  }

  std::shared_ptr<brpc::Channel> get_channel(const std::string& host) {
    {
      bthread::RWLockRdGuard rd_guard(instance_channel_map_mutex_);
      auto it = channels_.find(host);
      if (it != channels_.end()) {
        return it->second;
      }
    }
    bthread::RWLockWrGuard wr_guard(instance_channel_map_mutex_);
    auto it = channels_.find(host);
    if (it != channels_.end()) {
      return it->second;
    }

    brpc::ChannelOptions option;
    option.protocol = brpc::PROTOCOL_HTTP;
    option.connection_type = brpc::CONNECTION_TYPE_POOLED;
    option.max_retry = 3;
    auto channel = std::make_shared<brpc::Channel>();
    if (channel->Init(host.c_str(), &option) != 0) {
      LOG(ERROR) << "fail to init channel for " << host;
      return nullptr;
    }
    channels_[host] = channel;
    return channel;
  }

  bool download_data(const std::string& host,
                     const std::string& url,
                     std::string& data) {
    brpc::Controller cntl;
    cntl.http_request().uri() = url;
    cntl.set_timeout_ms(2000);
    auto channel = get_channel(host);
    if (!channel) {
      LOG(ERROR) << "Channel is null";
      return false;
    }
    channel->CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "Request failed: " << cntl.ErrorText();
      return false;
    }

    if (cntl.http_response().status_code() != 200) {
      LOG(ERROR) << "HTTP error: " << cntl.http_response().status_code();
      return false;
    }

    const butil::IOBuf& io = cntl.response_attachment();
    data = io.to_string();
    return true;
  }

  bool fetch_data(const std::string& url, std::string& data) {
    // parse url
    std::string host = parse_url(url);
    // fetch data
    return download_data(host, url, data);
  }

 private:
  bthread::RWLock instance_channel_map_mutex_;
  inline static std::unordered_map<std::string, std::shared_ptr<brpc::Channel>>
      channels_;
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

  bool process(const MMInputData& msg, MMInputItem& input) {
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
  virtual bool load(const MMInputData& msg, MMInputItem& input) = 0;
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
    return helper_.fetch_data(url, data);
  }

  std::string dataurl_prefix_{"data:image"};
  std::string httpurl_prefix_{"http"};

 private:
  FileDownloadHelper helper_;
};

class ImageHandler : public Handler {
 public:
  ImageHandler() {}

  virtual bool load(const proto::MMInputData& msg, MMInputItem& input) {
    input.clear();

    const auto& image_url = msg.image_url();
    const auto& url = image_url.url();

    if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
        0) {  // data url

      input.type_ = MMType::IMAGE;
      return this->load_from_dataurl(url, input.raw_data_);
    } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
               0) {  // http url

      input.type_ = MMType::IMAGE;
      return this->load_from_http(url, input.raw_data_);
    }
  }

  virtual bool load(const MMInputData& msg, MMInputItem& input) {
    input.clear();

    const auto& url = msg.image_url;
    if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
        0) {  // data url

      input.type_ = MMType::IMAGE;
      return this->load_from_dataurl(url, input.raw_data_);
    } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
               0) {  // http url

      input.type_ = MMType::IMAGE;
      return this->load_from_http(url, input.raw_data_);
    }
  }

  virtual bool decode(MMInputItem& input) {
    OpenCVImageDecoder decoder;
    return decoder.decode(input.raw_data_, input.decode_data_);
  }
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

  bool process(const std::string& type,
               const MMInputData& msg,
               MMInputItem& input) {
    auto itor = handlers_.find(type);
    if (itor == handlers_.end()) {
      return false;
    }

    auto& handler = itor->second;
    return handler->process(msg, input);
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<Handler>> handlers_;
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

bool MMInputHelper::trans(const std::vector<MMChatMessage>& raw_input_data,
                          std::vector<Message>& messages,
                          MMInputItemVec& inputs) {
  messages.clear();
  inputs.clear();
  messages.reserve(raw_input_data.size());
  inputs.reserve(raw_input_data.size());

  for (int idx = 0; idx < raw_input_data.size(); ++idx) {
    const auto& chat = raw_input_data[idx];
    const auto& role = chat.role;
    const auto& content = chat.content;

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

bool MMInputHelper::trans(const std::vector<MMInputData>& vec,
                          Message::MMContentVec& mmc,
                          MMInputItemVec& inputs) {
  mmc.clear();
  inputs.clear();

  for (int idx = 0; idx < vec.size(); ++idx) {
    const auto& item = vec[idx];
    const auto& type = item.type;

    if (type == "text") {
      mmc.emplace_back(type, item.text);
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
