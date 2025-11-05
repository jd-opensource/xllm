#include "rec_tokenizer.h"

#include <filesystem>

#include "common/version_singleton.h"
#include "state_dict/rec_vocab_dict.h"

namespace xllm {
RecTokenizer::RecTokenizer(const std::string_view& dir_path,
                           const TokenizerArgs& args) {
  args_ = args;
  dir_path_ = dir_path;
  model_version_ = std::filesystem::path(dir_path).filename();
}

bool RecTokenizer::encode(int64_t item_id,
                          std::vector<int32_t>* token_ids) const {
  if (!VersionSingleton<RecVocabDict>::GetInstance(model_version_)
           ->get_tokens_by_item(item_id, token_ids)) {
    return false;
  }

  return true;
}

bool RecTokenizer::decode(const Slice<int32_t>& token_ids,
                          bool skip_special_tokens,
                          std::vector<int64_t>* item_ids) const {
  CHECK_EQ(token_ids.size(), REC_TOKEN_SIZE);

  RecTokenTriple rec_token_triple;
  std::copy(token_ids.begin(), token_ids.end(), rec_token_triple.begin());

  if (!VersionSingleton<RecVocabDict>::GetInstance(model_version_)
           ->get_items_by_tokens(rec_token_triple, item_ids)) {
    return false;
  }

  return true;
}

size_t RecTokenizer::vocab_size() const {
  // currently, there is no voice size set in the tokenizer configuration. The
  // voice size can be obtained from the model args
  return 0;
}

std::unique_ptr<Tokenizer> RecTokenizer::clone() const {
  return std::make_unique<RecTokenizer>(dir_path_, args_);
}

}  // namespace xllm