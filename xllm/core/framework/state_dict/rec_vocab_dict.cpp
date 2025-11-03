#include "rec_vocab_dict.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>

#include "common/global_flags.h"
#include "util/timer.h"

namespace xllm {

bool RecVocabDict::initialize(const std::string& vocab_file) {
  if (initialized_) {
    return true;
  }

  Timer timer;

  if (vocab_file.empty()) {
    LOG(ERROR) << "content data file is empty, file: " << vocab_file;
    return false;
  }
  if (!std::filesystem::exists(vocab_file)) {
    LOG(ERROR) << "fail to find content data file: " << vocab_file;
    return false;
  }
  std::ifstream ifs(vocab_file.data(), std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    LOG(ERROR) << "fail to load content data file: " << vocab_file;
    return false;
  }

  const size_t file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  // each line of content : 1 * int64_t(item id) + REC_TOKEN_SIZE *
  //  int32_t(token id);
  const size_t itemid_size = sizeof(int64_t);
  const size_t tokens_size = REC_TOKEN_SIZE * sizeof(int32_t);
  const size_t line_size = tokens_size + itemid_size;
  const size_t estimated_lines = (file_size + line_size - 1) / line_size;

  // 2 and 4 are only empirical values
  item_to_tokens_map_.reserve(estimated_lines);
  tokens_to_items_map_.reserve(estimated_lines / 2);
  prefix_tokens_to_next_tokens_map_.reserve(estimated_lines / 4);

  int64_t item_id = 0;
  RecTokenTriple tokens;

  while (ifs.read(reinterpret_cast<char*>(&item_id), itemid_size) &&
         ifs.read(reinterpret_cast<char*>(tokens.data()), tokens_size)) {
    if (FLAGS_enable_constrained_decoding) {
      for (int i = 0; i < tokens.size(); i++) {
        std::vector<int32_t> prefix_tokens;

        for (int j = 0; j < i; j++) {
          prefix_tokens.emplace_back(tokens[j]);
        }

        prefix_tokens_to_next_tokens_map_[prefix_tokens].insert(tokens[i]);
      }
    }

    item_to_tokens_map_[item_id] = tokens;

    tokens_to_items_map_[tokens].emplace_back(item_id);
  }

  if (ifs.gcount() != 0 && ifs.gcount() != line_size) {
    LOG(ERROR) << "possibly containing incomplete lines : " << vocab_file;
    item_to_tokens_map_.clear();
    tokens_to_items_map_.clear();
    prefix_tokens_to_next_tokens_map_.clear();
    return false;
  }

  initialized_ = true;
  LOG(INFO) << "total line size:" << estimated_lines
            << ",parse tokens to item id map size: "
            << tokens_to_items_map_.size()
            << ", parse item to tokens map size:" << item_to_tokens_map_.size()
            << ", parse prefix tokens to next tokens map size:"
            << prefix_tokens_to_next_tokens_map_.size()
            << ", cost: " << timer.elapsed_seconds() << " seconds";

  return true;
}

bool RecVocabDict::get_items_by_tokens(const RecTokenTriple& rec_token_triple,
                                       std::vector<int64_t>* item_ids) const {
  CHECK_EQ(initialized_, true);
  CHECK_NE(item_ids, nullptr);

  auto iter = tokens_to_items_map_.find(rec_token_triple);
  if (iter == tokens_to_items_map_.end()) {
    return false;
  }

  std::copy(
      iter->second.begin(), iter->second.end(), std::back_inserter(*item_ids));

  return true;
}

bool RecVocabDict::get_tokens_by_item(int64_t item_id,
                                      std::vector<int32_t>* token_ids) const {
  CHECK_EQ(initialized_, true);
  CHECK_NE(token_ids, nullptr);

  auto iter = item_to_tokens_map_.find(item_id);
  if (iter == item_to_tokens_map_.end()) {
    return false;
  }

  std::copy(
      iter->second.begin(), iter->second.end(), std::back_inserter(*token_ids));

  return true;
}

const std::set<int32_t>& RecVocabDict::get_next_tokens_by_prefix_tokens(
    const Slice<int32_t>& prefix_token_ids) const {
  CHECK_EQ(initialized_, true);
  CHECK_LT(prefix_token_ids.size(), REC_TOKEN_SIZE);

  std::vector<int32_t> prefix_tokens_ids_vec = prefix_token_ids;
  auto iter = prefix_tokens_to_next_tokens_map_.find(prefix_tokens_ids_vec);
  if (iter == prefix_tokens_to_next_tokens_map_.end()) {
    static std::set<int32_t> empty_set;
    return empty_set;
  }

  return iter->second;
}

}  // namespace xllm