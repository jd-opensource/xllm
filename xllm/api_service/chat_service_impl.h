#pragma once

#include "api_service/api_service_impl.h"
#include "api_service/stream_call.h"
#include "chat.pb.h"
#include "multimodal.pb.h"

namespace xllm {

using ChatCall = StreamCall<proto::ChatRequest, proto::ChatResponse>;

// a class to handle completion requests
class ChatServiceImpl final : public APIServiceImpl<ChatCall> {
 public:
  ChatServiceImpl(LLMMaster* master, const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<ChatCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(ChatServiceImpl);

  const std::string parser_format_;
};

class VLMMaster;
using MMChatCall = StreamCall<proto::MMChatRequest, proto::ChatResponse>;

// a class to handle mm chat completion requests
class MMChatServiceImpl final {
 public:
  MMChatServiceImpl(VLMMaster* master, const std::vector<std::string>& models);
  MMChatServiceImpl() {}

  // brpc call_data needs to use shared_ptr
  void process_async(std::shared_ptr<MMChatCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(MMChatServiceImpl);

  VLMMaster* master_;
  absl::flat_hash_set<std::string> models_;
};

}  // namespace xllm
