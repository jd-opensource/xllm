/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "framework/tokenizer/fast_tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"

namespace xllm {

// Helper function to create a minimal valid tokenizer.json file for testing
// This creates a simple BPE tokenizer with a small vocabulary
std::string CreateTestTokenizerJson(const std::string& filepath) {
  // Minimal valid tokenizer.json for testing
  // This is a simplified BPE tokenizer configuration compatible with
  // HuggingFace tokenizers
  const std::string tokenizer_json = R"({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|bos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|eos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "hello", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 3, "content": "world", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 4, "content": "test", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
  ],
  "normalizer": {
    "type": "NFC"
  },
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "vocab": {
      "<|bos|>": 0,
      "<|eos|>": 1,
      "hello": 2,
      "world": 3,
      "test": 4,
      "h": 5,
      "e": 6,
      "l": 7,
      "o": 8,
      "w": 9,
      "r": 10,
      "d": 11,
      "t": 12,
      "s": 13,
      " ": 14
    },
    "merges": []
  }
})";

  std::ofstream file(filepath);
  if (!file.is_open()) {
    return "";
  }
  file << tokenizer_json;
  file.close();
  return filepath;
}

class FastTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary directory for test files
    test_dir_ = std::filesystem::temp_directory_path() / "fast_tokenizer_test";
    std::filesystem::create_directories(test_dir_);

    // Create test tokenizer.json file
    tokenizer_json_path_ = test_dir_ / "tokenizer.json";
    CreateTestTokenizerJson(tokenizer_json_path_.string());
  }

  void TearDown() override {
    // Clean up test files
    if (std::filesystem::exists(test_dir_)) {
      std::filesystem::remove_all(test_dir_);
    }
  }

  std::filesystem::path test_dir_;
  std::filesystem::path tokenizer_json_path_;
};

// Test that BOS token is added when add_bos_token is true
TEST_F(FastTokenizerTest, AddBosToken) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = false;

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // Check that BOS token (ID 0) is at the beginning
  EXPECT_EQ(ids[0], 0) << "BOS token should be at the beginning";

  // Verify that the rest of the tokens are present
  // The tokenizer should encode "hello world" into token IDs
  EXPECT_GT(ids.size(), 1) << "Should have more than just BOS token";
}

// Test that EOS token is added when add_eos_token is true
TEST_F(FastTokenizerTest, AddEosToken) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = false;
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // Check that EOS token (ID 1) is at the end
  EXPECT_EQ(ids.back(), 1) << "EOS token should be at the end";

  // Verify that the rest of the tokens are present
  EXPECT_GT(ids.size(), 1) << "Should have more than just EOS token";
}

// Test that both BOS and EOS tokens are added when both flags are true
TEST_F(FastTokenizerTest, AddBothBosAndEosTokens) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_GE(ids.size(), 2) << "Should have at least BOS and EOS tokens";

  // Check that BOS token (ID 0) is at the beginning
  EXPECT_EQ(ids[0], 0) << "BOS token should be at the beginning";

  // Check that EOS token (ID 1) is at the end
  EXPECT_EQ(ids.back(), 1) << "EOS token should be at the end";

  // Verify that there are tokens in between
  EXPECT_GT(ids.size(), 2) << "Should have tokens between BOS and EOS";
}

// Test that no special tokens are added when both flags are false
TEST_F(FastTokenizerTest, NoSpecialTokens) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = false;
  args.add_eos_token() = false;

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // Check that BOS token (ID 0) is NOT at the beginning
  EXPECT_NE(ids[0], 0) << "BOS token should not be present";

  // Check that EOS token (ID 1) is NOT at the end
  EXPECT_NE(ids.back(), 1) << "EOS token should not be present";
}

// Test that BOS token is not added when bos_token is empty
TEST_F(FastTokenizerTest, AddBosTokenWithEmptyToken) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "";  // Empty token
  args.add_eos_token() = false;

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // BOS token should not be added because bos_token is empty
  EXPECT_NE(ids[0], 0)
      << "BOS token should not be added when bos_token is empty";
}

// Test that EOS token is not added when eos_token is empty
TEST_F(FastTokenizerTest, AddEosTokenWithEmptyToken) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = false;
  args.add_eos_token() = true;
  args.eos_token() = "";  // Empty token

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // EOS token should not be added because eos_token is empty
  EXPECT_NE(ids.back(), 1)
      << "EOS token should not be added when eos_token is empty";
}

// Test that BOS token is not duplicated when it already exists
// This simulates the case where the underlying tokenizer already added BOS
TEST_F(FastTokenizerTest, SkipBosTokenWhenAlreadyPresent) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = false;

  FastTokenizer tokenizer(args);

  // First encode a text that doesn't start with BOS
  std::vector<int32_t> ids1;
  bool success1 = tokenizer.encode("hello world", &ids1);
  ASSERT_TRUE(success1);
  ASSERT_FALSE(ids1.empty());

  // Verify BOS token was added
  EXPECT_EQ(ids1[0], 0) << "BOS token should be added";
  size_t size_with_bos = ids1.size();

  // Now encode text that starts with BOS token directly
  // We'll encode the BOS token itself, which should result in BOS being the
  // first token
  std::vector<int32_t> ids2;
  bool success2 = tokenizer.encode("<|bos|> hello world", &ids2);
  ASSERT_TRUE(success2);
  ASSERT_FALSE(ids2.empty());

  // The BOS token should be present, but we should not have added it twice
  // Count how many times BOS token (ID 0) appears at the beginning
  int bos_count_at_start = 0;
  for (size_t i = 0; i < ids2.size() && ids2[i] == 0; ++i) {
    bos_count_at_start++;
  }

  // Should have at most one BOS token at the beginning
  EXPECT_LE(bos_count_at_start, 1)
      << "BOS token should not be duplicated when already present";
}

// Test that EOS token is not duplicated when it already exists
TEST_F(FastTokenizerTest, SkipEosTokenWhenAlreadyPresent) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = false;
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);

  // First encode a text that doesn't end with EOS
  std::vector<int32_t> ids1;
  bool success1 = tokenizer.encode("hello world", &ids1);
  ASSERT_TRUE(success1);
  ASSERT_FALSE(ids1.empty());

  // Verify EOS token was added
  EXPECT_EQ(ids1.back(), 1) << "EOS token should be added";
  size_t size_with_eos = ids1.size();

  // Now encode text that ends with EOS token directly
  std::vector<int32_t> ids2;
  bool success2 = tokenizer.encode("hello world <|eos|>", &ids2);
  ASSERT_TRUE(success2);
  ASSERT_FALSE(ids2.empty());

  // The EOS token should be present, but we should not have added it twice
  // Count how many times EOS token (ID 1) appears at the end
  int eos_count_at_end = 0;
  for (int i = ids2.size() - 1; i >= 0 && ids2[i] == 1; --i) {
    eos_count_at_end++;
  }

  // Should have at most one EOS token at the end
  EXPECT_LE(eos_count_at_end, 1)
      << "EOS token should not be duplicated when already present";
}

// Test that both BOS and EOS tokens are not duplicated when both already exist
TEST_F(FastTokenizerTest, SkipBothBosAndEosTokensWhenAlreadyPresent) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);

  // Encode text that already contains both BOS and EOS tokens
  std::vector<int32_t> ids;
  bool success = tokenizer.encode("<|bos|> hello world <|eos|>", &ids);
  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // Count BOS tokens at the beginning
  int bos_count_at_start = 0;
  for (size_t i = 0; i < ids.size() && ids[i] == 0; ++i) {
    bos_count_at_start++;
  }

  // Count EOS tokens at the end
  int eos_count_at_end = 0;
  for (int i = ids.size() - 1; i >= 0 && ids[i] == 1; --i) {
    eos_count_at_end++;
  }

  // Should have at most one BOS token at the beginning
  EXPECT_LE(bos_count_at_start, 1)
      << "BOS token should not be duplicated when already present";

  // Should have at most one EOS token at the end
  EXPECT_LE(eos_count_at_end, 1)
      << "EOS token should not be duplicated when already present";
}

}  // namespace xllm
