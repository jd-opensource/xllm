#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "chat.pb.h"
#include "chat_service_impl.h"
#include "core/framework/request/request_output.h"
#include "function_call/function_call_parser.h"

namespace xllm {

class StreamingFunctionCallHandler {
 public:
  StreamingFunctionCallHandler(
      const std::vector<function_call::JsonTool>& tools,
      const std::string& parser_format);

  ~StreamingFunctionCallHandler() = default;

  template <typename ChatCall>
  bool process_streaming_output(std::shared_ptr<ChatCall> call,
                                bool include_usage,
                                std::unordered_set<size_t>* first_message_sent,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const RequestOutput& output);

 private:
  std::unique_ptr<function_call::FunctionCallParser> parser_;

  std::vector<function_call::JsonTool> tools_;
  std::string parser_format_;

  std::unordered_map<size_t, bool> has_tool_calls_;

  template <typename ChatCall>
  bool process_tool_call_stream(std::shared_ptr<ChatCall> call,
                                size_t index,
                                const std::string& delta,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model);

  template <typename ChatCall>
  bool check_for_unstreamed_tool_args(std::shared_ptr<ChatCall> call,
                                      size_t index,
                                      const std::string& request_id,
                                      int64_t created_time,
                                      const std::string& model);

  template <typename ChatCall>
  bool send_tool_call_chunk(std::shared_ptr<ChatCall> call,
                            size_t index,
                            const std::string& tool_call_id,
                            const std::string& function_name,
                            const std::string& arguments,
                            int tool_index,
                            const std::string& request_id,
                            int64_t created_time,
                            const std::string& model);

  template <typename ChatCall>
  bool send_normal_text_chunk(std::shared_ptr<ChatCall> call,
                              size_t index,
                              const std::string& content,
                              const std::string& request_id,
                              int64_t created_time,
                              const std::string& model);
};

}  // namespace xllm