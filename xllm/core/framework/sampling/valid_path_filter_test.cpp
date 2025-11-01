#include "valid_path_filter.h"

#include <ATen/ops/equal.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <fstream>
#include <iostream>

namespace xllm {

TEST(ValidPathFilterTest, Vector) {
  // 基于实际使用场景的测试数据
  // tokens_list表示有效的token序列路径，每个序列长度为3
  std::vector<std::vector<int32_t>> tokens_list = {
      {1, 2, 3},  // 序列1: 1->2->3
      {1, 2, 4},  // 序列2: 1->2->4
      {1, 3, 5},  // 序列3: 1->3->5
      {2, 4, 6},  // 序列4: 2->4->6
      {3, 5, 7}   // 序列5: 3->5->7
  };

  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  int32_t vocab_size = 8;  // 词汇表大小为8 (tokens 0-7)

  ValidPathFilter filter =
      ValidPathFilter(tokens_list, vocab_size, dtype, device);

  // 测试不同的候选token序列
  std::vector<std::vector<int32_t>> candidate_tokens = {
      {1, 2},  // 前缀[1,2]，应该允许token 3和4
      {1},     // 前缀[1]，应该允许token 2和3
      {},      // 空前缀，应该允许第一个token 1,2,3
      {2, 4},  // 前缀[2,4]，应该允许token 6
      {9, 9}   // 无效前缀，应该全部被mask
  };

  const auto options = torch::dtype(dtype).device(device);
  torch::Tensor mask = filter.forward(candidate_tokens);

  // 验证输出形状
  EXPECT_EQ(mask.sizes(),
            torch::IntArrayRef({candidate_tokens.size(), vocab_size}));

  // 验证mask值
  // mask值为0表示允许，-10000表示禁止

  // 对于前缀[1,2]：下一个token可以是3或4
  auto mask_1_2 = mask[0];
  EXPECT_EQ(mask_1_2[3].item<float>(), 0.0f);       // token 3允许
  EXPECT_EQ(mask_1_2[4].item<float>(), 0.0f);       // token 4允许
  EXPECT_EQ(mask_1_2[0].item<float>(), -10000.0f);  // token 0禁止
  EXPECT_EQ(mask_1_2[1].item<float>(), -10000.0f);  // token 1禁止
  EXPECT_EQ(mask_1_2[2].item<float>(), -10000.0f);  // token 2禁止

  // 对于前缀[1]：下一个token可以是2或3
  auto mask_1 = mask[1];
  EXPECT_EQ(mask_1[2].item<float>(), 0.0f);       // token 2允许
  EXPECT_EQ(mask_1[3].item<float>(), 0.0f);       // token 3允许
  EXPECT_EQ(mask_1[0].item<float>(), -10000.0f);  // token 0禁止
  EXPECT_EQ(mask_1[1].item<float>(), -10000.0f);  // token 1禁止

  // 对于空前缀[]：第一个token可以是1,2,3
  auto mask_empty = mask[2];
  EXPECT_EQ(mask_empty[1].item<float>(), 0.0f);       // token 1允许
  EXPECT_EQ(mask_empty[2].item<float>(), 0.0f);       // token 2允许
  EXPECT_EQ(mask_empty[3].item<float>(), 0.0f);       // token 3允许
  EXPECT_EQ(mask_empty[0].item<float>(), -10000.0f);  // token 0禁止
}

TEST(ValidPathFilterTest, File) {
  // 创建测试数据文件
  std::vector<std::vector<int32_t>> tokens_list = {
      {1, 2, 3}, {1, 2, 4}, {1, 3, 5}, {2, 4, 6}, {3, 5, 7}};

  const std::string rec_tokens_file = "./test_data.bin";

  // 清理旧文件
  if (std::ifstream(rec_tokens_file)) {
    std::remove(rec_tokens_file.c_str());
  }

  // 按照实现期望的格式写入文件：int64_t item_id + 3个int32_t
  std::ofstream outfile(rec_tokens_file, std::ios::binary);
  if (!outfile) {
    LOG(ERROR) << "Failed to create test file: " << rec_tokens_file;
    return;
  }

  int64_t item_id = 0;
  for (const auto& row : tokens_list) {
    outfile.write(reinterpret_cast<const char*>(&item_id), sizeof(int64_t));
    outfile.write(reinterpret_cast<const char*>(row.data()),
                  row.size() * sizeof(int32_t));
    item_id++;
  }
  outfile.close();

  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  int32_t vocab_size = 8;

  ValidPathFilter filter =
      ValidPathFilter(rec_tokens_file, vocab_size, dtype, device);

  // 使用相同的测试用例
  std::vector<std::vector<int32_t>> candidate_tokens = {
      {1, 2},  // 前缀[1,2]
      {1},     // 前缀[1]
      {}       // 空前缀
  };

  const auto options = torch::dtype(dtype).device(device);
  torch::Tensor mask = filter.forward(candidate_tokens);

  // 验证输出形状
  EXPECT_EQ(mask.sizes(),
            torch::IntArrayRef({candidate_tokens.size(), vocab_size}));

  // 验证与Vector测试相同的结果
  // 对于前缀[1,2]：下一个token可以是3或4
  auto mask_1_2 = mask[0];
  EXPECT_EQ(mask_1_2[3].item<float>(), 0.0f);
  EXPECT_EQ(mask_1_2[4].item<float>(), 0.0f);

  // 对于前缀[1]：下一个token可以是2或3
  auto mask_1 = mask[1];
  EXPECT_EQ(mask_1[2].item<float>(), 0.0f);
  EXPECT_EQ(mask_1[3].item<float>(), 0.0f);

  // 对于空前缀[]：第一个token可以是1,2,3
  auto mask_empty = mask[2];
  EXPECT_EQ(mask_empty[1].item<float>(), 0.0f);
  EXPECT_EQ(mask_empty[2].item<float>(), 0.0f);
  EXPECT_EQ(mask_empty[3].item<float>(), 0.0f);

  // 清理测试文件
  if (std::ifstream(rec_tokens_file)) {
    std::remove(rec_tokens_file.c_str());
  }
}

TEST(ValidPathFilterTest, EmptyInput) {
  // 测试空输入的情况
  std::vector<std::vector<int32_t>> tokens_list = {{1, 2, 3}};
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  int32_t vocab_size = 5;

  ValidPathFilter filter =
      ValidPathFilter(tokens_list, vocab_size, dtype, device);

  // 测试空的候选token列表
  std::vector<std::vector<int32_t>> empty_candidates = {};
  torch::Tensor mask = filter.forward(empty_candidates);

  // 应该返回未定义的tensor
  EXPECT_FALSE(mask.defined());
}

}  // namespace xllm
