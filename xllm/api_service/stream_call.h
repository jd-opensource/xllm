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

#include <brpc/controller.h>
#include <butil/iobuf.h>
#include <glog/logging.h>
#include <json2pb/pb_to_json.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>

#include "api_service/call.h"
#include "core/common/types.h"

namespace xllm {

template <typename Request, typename Response>
class StreamCall : public CallImpl<Request, Response> {
 public:
  using ReqType = Request;
  using ResType = Response;

  StreamCall(brpc::Controller* controller,
             ::google::protobuf::Closure* done,
             Request* request,
             Response* response,
             bool use_arena = true)
      : CallImpl<Request, Response>(controller,
                                    done,
                                    request,
                                    response,
                                    use_arena) {
    stream_ = this->request_->stream();
    if (stream_) {
      pa_ = this->controller_->CreateProgressiveAttachment();

      // Send the first SSE response
      this->controller_->http_response().set_content_type("text/event-stream");
      this->controller_->http_response().set_status_code(200);
      this->controller_->http_response().SetHeader("Connection", "keep-alive");
      this->controller_->http_response().SetHeader("Cache-Control", "no-cache");
      // Done Run first for steam response
      this->done_->Run();

    } else {
      this->controller_->http_response().SetHeader(
          "Content-Type", "text/javascript; charset=utf-8");
    }

    json_options_.bytes_to_base64 = false;
    json_options_.jsonify_empty_array = true;
  }

  ~StreamCall() override {
    // For non stream response, call brpc done Run
    if (!stream_) {
      this->done_->Run();
    }
  }

  bool write_and_finish(Response& response) {
    butil::IOBufAsZeroCopyOutputStream json_output(
        &this->controller_->response_attachment());
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      return finish_with_error(StatusCode::UNKNOWN, err_msg);
    }
    return true;
  }

  bool finish_with_error(const StatusCode& code,
                         const std::string& error_message) {
    if (!stream_) {
      this->controller_->SetFailed(error_message);

    } else {
      io_buf_.clear();
      io_buf_.append(error_message);
      pa_->Write(io_buf_);
    }

    return true;
  }

  // For stream response
  bool write(Response& response) {
    io_buf_.clear();
    io_buf_.append("data: ");
    butil::IOBufAsZeroCopyOutputStream json_output(&io_buf_);
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      LOG(ERROR) << "Failed to convert proto to json: " << err_msg;
      return false;
    }
    io_buf_.append("\n\n");

    connection_status_ |= pa_->Write(io_buf_);
    return true;
  }

  // For stream response
  bool finish() {
    io_buf_.clear();
    io_buf_.append("data: [DONE]\n\n");

    pa_->Write(io_buf_);
    return true;
  }

  bool is_disconnected() const override {
    if (stream_) {
      return connection_status_ != 0;
    } else {
      if (this->controller_) {
        return this->controller_->IsCanceled();
      }
      return true;
    }
  }

 private:
  bool stream_ = false;

  butil::intrusive_ptr<brpc::ProgressiveAttachment> pa_;
  butil::IOBuf io_buf_;

  json2pb::Pb2JsonOptions json_options_;

  int connection_status_ = 0;
};

}  // namespace xllm
