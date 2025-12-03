#include <unistd.h>

#include <iostream>

#include "llm.h"

std::string devices = "npu:0";
std::string model_path = "/export/home/models/Qwen3-8B";

int main(int argc, char** argv) {
  xllm::LLM llm;
  xllm::XLLM_InitLLMOptions options;
  bool ret = llm.Initialize(model_path, devices, options);
  if (!ret) {
    std::cout << "llm init failed" << std::endl;
    return -1;
  }

  std::cout << "llm init succefully" << std::endl;

  {
    std::cout << "llm completions start" << std::endl;

    std::string prompt =
        "请推荐3个便宜又好用的电动剃须刀，简要说下产品名称、价格、特色即可";
    xllm::XLLM_RequestParams params;
    params.max_tokens = 500;
    xllm::XLLM_Response response =
        llm.Completions("Qwen3-8B", prompt, 20000, params);

    if (response.status_code != xllm::XLLM_StatusCode::kSuccess) {
      std::cout << "llm completions failed, error info:" << response.error_info
                << std::endl;
      return -1;
    } else {
      for (auto choice : response.choices) {
        if (choice.text.has_value()) {
          std::cout << "llm completions output:" << choice.text.value().c_str()
                    << std::endl;
        }
      }
    }

    std::cout << "llm completions end" << std::endl;
  }

  {
    std::cout << "llm chat completions start" << std::endl;
    xllm::XLLM_ChatMessage message;
    message.role = "user";
    message.content =
        "你是一个电商场景专家，当前场景是京东B商城电商搜索引擎，B商城是服务于企"
        "业用户的企业版京东商城，经营品类较全面，任务是判断在电商搜索引擎中'"
        "用户query'和'商品标题'是否相关。\n判别标准：如果搜索'用户query'返回'"
        "商品标题'符合用户的需求则任务相关。\n输出要求：你在'相关'或者'不相关'"
        "中给答案，不要说其他内容,用户query:火锅酱料\n商品标题:"
        "草原红太阳火锅底料蘸料多味烧烤酱料番茄酱韭花酱夜宵搭配 "
        "【新】香辣味烧烤酱100g";

    std::vector<xllm::XLLM_ChatMessage> messages;
    messages.emplace_back(message);
    xllm::XLLM_RequestParams params;
    params.max_tokens = 100;

    xllm::XLLM_Response response =
        llm.ChatCompletions("Qwen3-8B", messages, 20000, params);

    if (response.status_code != xllm::XLLM_StatusCode::kSuccess) {
      std::cout << "llm completions failed, error info:" << response.error_info
                << std::endl;
      return -1;
    } else {
      for (auto choice : response.choices) {
        if (choice.message.has_value()) {
          std::cout << "llm completions output: role: "
                    << choice.message.value().role.c_str()
                    << ",content: " << choice.message.value().content.c_str()
                    << std::endl;
        }
      }
    }

    std::cout << "llm chat completions end" << std::endl;
  }

  sleep(10);

  return 0;
}