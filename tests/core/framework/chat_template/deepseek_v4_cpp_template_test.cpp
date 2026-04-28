/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/chat_template/deepseek_v4_cpp_template.h"

#include <gtest/gtest.h>

namespace xllm {

TEST(DeepseekV4CppTemplate, BasicUserMessage) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV4CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "hello");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("<｜begin▁of▁sentence｜>"), std::string::npos);
  EXPECT_NE(prompt->find("<｜User｜>hello"), std::string::npos);
  // V4: Assistant marker appended via transition
  EXPECT_NE(prompt->find("<｜Assistant｜>"), std::string::npos);
}

TEST(DeepseekV4CppTemplate, DefaultThinkingModeIsChat) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV4CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "hello");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("</think>"), std::string::npos);
  EXPECT_EQ(prompt->find("<think>"), std::string::npos);
}

TEST(DeepseekV4CppTemplate, ThinkingModeEnabledByKwargs) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV4CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "hello");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  kwargs["thinking"] = true;

  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("<think>"), std::string::npos);
}

TEST(DeepseekV4CppTemplate, ToolsInjectionDsmlFormat) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV4CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "weather?");

  std::vector<JsonTool> tools;
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "get_weather";
  tool.function.description = "query weather";
  tool.function.parameters = nlohmann::json{
      {"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}};
  tools.push_back(tool);

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, tools, kwargs);
  ASSERT_TRUE(prompt.has_value());

  // V4 DSML format with tool_calls block
  EXPECT_NE(prompt->find("## Tools"), std::string::npos);
  EXPECT_NE(prompt->find("get_weather"), std::string::npos);
  EXPECT_NE(prompt->find("tool_calls"), std::string::npos);
  EXPECT_NE(prompt->find("Available Tool Schemas"), std::string::npos);
}

TEST(DeepseekV4CppTemplate, ToolMessagesAreMergedIntoUser) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV4CppTemplate encoder(args);

  // user -> assistant(tool_calls) -> tool
  ChatMessages messages;
  messages.emplace_back("user", "weather?");

  Message assistant_msg("assistant", "calling");
  Message::ToolCall tc;
  tc.id = "1";
  tc.type = "function";
  tc.function.name = "get_weather";
  tc.function.arguments = R"({"city":"beijing"})";
  assistant_msg.tool_calls = Message::ToolCallVec{tc};
  messages.push_back(assistant_msg);

  Message tool_msg("tool", "sunny");
  tool_msg.tool_call_id = "1";
  messages.push_back(tool_msg);

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  // Tool output rendered as <tool_result> within user
  EXPECT_NE(prompt->find("<tool_result>sunny</tool_result>"),
            std::string::npos);
  // No raw "tool" role should exist
  // (merged into user via content_blocks)
}

TEST(DeepseekV4CppTemplate, DropThinkingDisabledWhenToolsExist) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV4CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "hello");

  std::vector<JsonTool> tools;
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "search";
  tool.function.description = "search";
  tool.function.parameters = nlohmann::json{{"type", "object"}};
  tools.push_back(tool);

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  kwargs["thinking"] = true;

  auto prompt = encoder.apply(messages, tools, kwargs);
  ASSERT_TRUE(prompt.has_value());

  // With tools present, drop_thinking is disabled
  // so <think> should appear (thinking mode on)
  EXPECT_NE(prompt->find("<think>"), std::string::npos);
}

}  // namespace xllm
