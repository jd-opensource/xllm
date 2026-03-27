/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "sampler_utils.h"

#include <absl/strings/ascii.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <glog/logging.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/global_flags.h"

namespace xllm {
namespace {

constexpr std::string kDefaultQwen3RerankerCandidateTokenIds = "2152,9693";

std::vector<std::string> parse_csv_entries(const std::string& raw_value,
                                           const std::string& flag_name) {
  std::vector<std::string> entries;
  for (absl::string_view value : absl::StrSplit(raw_value, ',')) {
    std::string entry(value);
    absl::StripAsciiWhitespace(&entry);
    if (entry.empty()) {
      LOG(FATAL) << "Flag --" << flag_name
                 << " contains empty entries. Raw value: " << raw_value;
    }
    entries.push_back(std::move(entry));
  }
  if (entries.empty()) {
    LOG(FATAL) << "Flag --" << flag_name << " must not be empty when set.";
  }
  return entries;
}

void validate_token_id_range(int64_t token_id,
                             int64_t vocab_size,
                             const std::string& source,
                             const std::string& flag_name) {
  if (token_id < 0 || token_id >= vocab_size) {
    LOG(FATAL) << "Resolved token id " << token_id << " from --" << flag_name
               << " entry [" << source << "] is out of vocab range [0, "
               << (vocab_size - 1) << "].";
  }
}

void maybe_validate_candidate_set_size(const std::vector<int64_t>& token_ids) {
  if (!token_ids.empty() && token_ids.size() < 2) {
    LOG(FATAL) << "Candidate set size must be at least 2 after deduplication. "
               << "Current unique candidate count: " << token_ids.size();
  }
  if (FLAGS_enable_qwen3_reranker && token_ids.empty()) {
    LOG(FATAL) << "Flag --enable_qwen3_reranker=true requires configuring "
                  "--candidate_token_ids.";
  }
  if (FLAGS_enable_qwen3_reranker && token_ids.size() != 2) {
    LOG(FATAL) << "Flag --enable_qwen3_reranker=true requires exactly two "
                  "unique candidates. Current unique candidate count: "
               << token_ids.size();
  }
}

}  // namespace

std::vector<int64_t> resolve_candidate_token_ids(int64_t vocab_size) {
  if (FLAGS_enable_qwen3_reranker && FLAGS_candidate_token_ids.empty()) {
    FLAGS_candidate_token_ids = kDefaultQwen3RerankerCandidateTokenIds;
    LOG(INFO) << "Flag --enable_qwen3_reranker=true and --candidate_token_ids "
                 "is unset; fallback to default candidate token ids: "
              << FLAGS_candidate_token_ids;
  }

  const bool has_candidate_token_ids = !FLAGS_candidate_token_ids.empty();
  if (!has_candidate_token_ids) {
    return {};
  }
  if (vocab_size <= 0) {
    LOG(FATAL) << "Invalid vocab size for candidate token resolution: "
               << vocab_size;
  }

  auto entries =
      parse_csv_entries(FLAGS_candidate_token_ids, "candidate_token_ids");

  std::vector<int64_t> resolved_token_ids;
  std::unordered_set<int64_t> seen_token_ids;
  resolved_token_ids.reserve(entries.size());
  seen_token_ids.reserve(entries.size());

  for (const auto& entry : entries) {
    int64_t token_id = 0;
    if (!absl::SimpleAtoi(entry, &token_id)) {
      LOG(FATAL) << "Flag --candidate_token_ids contains a non-integer value: "
                 << entry;
    }
    validate_token_id_range(token_id, vocab_size, entry, "candidate_token_ids");
    if (seen_token_ids.insert(token_id).second) {
      resolved_token_ids.push_back(token_id);
    }
  }

  maybe_validate_candidate_set_size(resolved_token_ids);
  return resolved_token_ids;
}

}  // namespace xllm
