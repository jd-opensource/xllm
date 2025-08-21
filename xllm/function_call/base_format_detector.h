#pragma once

#include <glog/logging.h>

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "chat.pb.h"
#include "core_types.h"
#include "utils.h"

namespace xllm {
namespace function_call {

class BaseFormatDetector {
 public:
  BaseFormatDetector();
  virtual ~BaseFormatDetector() = default;

  BaseFormatDetector(const BaseFormatDetector&) = delete;
  BaseFormatDetector& operator=(const BaseFormatDetector&) = delete;

  std::unordered_map<std::string, int> get_tool_indices(
      const std::vector<JsonTool>& tools);

  std::vector<ToolCallItem> parse_base_json(const nlohmann::json& json_obj,
                                            const std::vector<JsonTool>& tools);

  virtual StreamingParseResult detect_and_parse(
      const std::string& text,
      const std::vector<JsonTool>& tools) = 0;

  virtual bool has_tool_call(const std::string& text) = 0;

  virtual StreamingParseResult parse_streaming_increment(
      const std::string& new_text,
      const std::vector<JsonTool>& tools);

  std::vector<std::unordered_map<std::string, std::string>> prev_tool_call_arr_;

  std::vector<std::string> streamed_args_for_tool_;

 protected:
  std::string buffer_;

  int current_tool_id_;

  bool current_tool_name_sent_;

  std::string bot_token_;
  std::string eot_token_;
  std::string tool_call_separator_;

  int _ends_with_partial_token(const std::string& buffer,
                               const std::string& bot_token) const;

  std::unordered_map<std::string, int> tool_indices_;
};

}  // namespace function_call
}  // namespace xllm