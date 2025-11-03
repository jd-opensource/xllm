#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer.h"
#include "tokenizer_args.h"
#include "util/slice.h"

namespace xllm {

class RecTokenizer : public Tokenizer {
 public:
  RecTokenizer(const std::string_view& dir_path, const TokenizerArgs& args);

  virtual ~RecTokenizer() = default;

  bool encode(int64_t item_id, std::vector<int32_t>* token_ids) const override;

  bool decode(const Slice<int32_t>& token_ids,
              bool skip_special_tokens,
              std::vector<int64_t>* item_ids) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  TokenizerArgs args_;

  std::string dir_path_;

  std::string model_version_;
};

}  // namespace xllm
