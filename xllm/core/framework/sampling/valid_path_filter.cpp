#include "valid_path_filter.h"

#include <glog/logging.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>

#include "util/env_var.h"
#include "util/hash_util.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/timer.h"

namespace xllm {

namespace {

void parse_valid_path_filter_file(
    std::vector<std::vector<int32_t>>& tokens_list,
    const std::string& valid_path_filter_file) {
  if (valid_path_filter_file.empty()) {
    LOG(WARNING) << "Get empty vaild path filter file: "
                 << valid_path_filter_file;
    return;
  }
  if (!std::filesystem::exists(valid_path_filter_file)) {
    LOG(ERROR) << "Failed to find vaild path filter file: "
               << valid_path_filter_file;
    return;
  }
  std::ifstream ifs(valid_path_filter_file, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    LOG(ERROR) << "Failed to load vaild path filter file: "
               << valid_path_filter_file;
    return;
  }

  const size_t file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  const int elements_per_line = 3;
  const size_t elements_size = elements_per_line * sizeof(int32_t);
  const size_t line_size = sizeof(int64_t) + elements_size;
  const size_t estimated_lines = (file_size + line_size - 1) / line_size;

  tokens_list.reserve(estimated_lines);

  int64_t item_id;
  std::vector<int32_t> buffer(elements_per_line);
  while (ifs.read(reinterpret_cast<char*>(&item_id), sizeof(int64_t)) &&
         ifs.read(reinterpret_cast<char*>(buffer.data()), elements_size)) {
    tokens_list.emplace_back(buffer.begin(), buffer.end());
  }
  LOG(INFO) << "ValidPathFilter parse tokens list size:" << tokens_list.size();

  if (ifs.gcount() != 0 && ifs.gcount() != line_size) {
    LOG(ERROR) << "Possibly containing incomplete lines : "
               << valid_path_filter_file;
    return;
  }
}
}  // namespace

float ValidPathFilter::pre_mask_factor_ = -10000.0f;

ValidPathFilter::ValidPathFilter(const std::string valid_path_filter_file,
                                 const int32_t vocab_size,
                                 torch::ScalarType dtype,
                                 torch::Device device)
    : vocab_size_(vocab_size), dtype_(dtype), device_(device) {
  std::vector<std::vector<int32_t>> tokens_list;
  Timer timer;
  parse_valid_path_filter_file(tokens_list, valid_path_filter_file);
  init_cached_mask(tokens_list, vocab_size);
  LOG(INFO) << " ValidPathFilter generate " << cached_sparse_mask_.size()
            << " key for " << tokens_list.size() << " items which took "
            << timer.elapsed_seconds() << " secs.";
}

ValidPathFilter::ValidPathFilter(
    const std::vector<std::vector<int32_t>>& tokens_list,
    const int32_t vocab_size,
    torch::ScalarType dtype,
    torch::Device device)
    : vocab_size_(vocab_size), dtype_(dtype), device_(device) {
  init_cached_mask(tokens_list, vocab_size);
}

void ValidPathFilter::init_cached_mask(
    const std::vector<std::vector<int32_t>>& tokens_list,
    const int32_t vocab_size) {
  size_t total_num = tokens_list.size();
  if (total_num > 0) {
    init_cached_tokens_ = true;
  }

  // init extra thread pool
  thread_num_ = util::get_int_env(util::EXTRA_THREAD_NUM, 16);
  extra_threadpool_ = std::make_unique<ThreadPool>(thread_num_);

  // generate mask
  torch::TensorOptions options = torch::dtype(dtype_).device(device_);
  first_token_mask_ = torch::full({vocab_size}, pre_mask_factor_, dtype_);
  empty_place_holder_ = torch::full({vocab_size}, 0.0f, options);

  cached_sparse_mask_.reserve(total_num);
  for (size_t t_idx = 0; t_idx < total_num; t_idx++) {
    Slice<int32_t> tokens_slice(tokens_list[t_idx]);
    CHECK_EQ(tokens_slice.size(), 3);

    // handle first token
    first_token_mask_[tokens_slice[0]] = 0;

    // handle extra token
    for (int i = 1; i < tokens_slice.size(); i++) {
      Murmur3Key murmur3_key;
      Slice<int32_t> sub_slice(tokens_slice.data(), i);
      murmur_hash3(nullptr, sub_slice, murmur3_key.data);
      auto iter = cached_sparse_mask_.find(murmur3_key);
      if (iter != cached_sparse_mask_.end()) {
        iter->second.push_back(tokens_slice[i]);
      } else {
        std::vector<int32_t> false_indices = {tokens_slice[i]};
        cached_sparse_mask_.emplace(std::make_pair(murmur3_key, false_indices));
      }
    }
  }

  // Remove duplicates and sort for better performance
  // Sort false indices in sparse masks for better performance
  for (auto& pair : cached_sparse_mask_) {
    std::sort(pair.second.begin(), pair.second.end());
    pair.second.erase(std::unique(pair.second.begin(), pair.second.end()),
                      pair.second.end());
  }
  // first_token_mask_ = safe_to(first_token_mask_, device_, true);
  LOG(INFO) << " ValidPathFilter third sparse storage: "
            << cached_sparse_mask_.size();
}

torch::Tensor ValidPathFilter::forward(
    const std::vector<std::vector<int32_t>>& tokens_list) {
  if (!init_cached_tokens_ || tokens_list.size() == 0) {
    return torch::Tensor();
  }

  size_t token_size = tokens_list[0].size();

  // prepare mask for first token
  if (token_size == 0) {
    size_t total_nums = tokens_list.size();
    auto mask = first_token_mask_.unsqueeze(0);
    return mask.repeat({total_nums, 1});
  }
  return forward_sparse_mask(tokens_list);
}

torch::Tensor ValidPathFilter::forward_sparse_mask(
    const std::vector<std::vector<int32_t>>& tokens_list) {
  Timer timer;
  size_t total_nums = tokens_list.size();
  torch::TensorOptions options = torch::dtype(dtype_).device(device_);
  auto mask = torch::full({total_nums, vocab_size_}, pre_mask_factor_, options);

  // Global batch collection for sparse storage optimization
  std::vector<int64_t> global_batch_token_indices;
  std::vector<int64_t> global_batch_vocab_indices;
  std::mutex batch_mutex;  // Protect global batch vectors in multi-threading

  // Pre-allocate space: assume max 8192 false indices per token
  global_batch_token_indices.reserve(8192 * total_nums);
  global_batch_vocab_indices.reserve(8192 * total_nums);

  auto update_mask = [&](size_t start_idx, size_t end_idx) {
    // Local collection for this thread
    std::vector<int64_t> local_token_indices;
    std::vector<int64_t> local_vocab_indices;
    local_token_indices.reserve(8192 * (end_idx - start_idx));
    local_vocab_indices.reserve(8192 * (end_idx - start_idx));

    for (size_t token_idx = start_idx; token_idx < end_idx; ++token_idx) {
      auto& tokens = tokens_list[token_idx];
      if (tokens.size() == 0) {
        mask[token_idx] = first_token_mask_.to(device_);
      } else {
        Slice<int32_t> tokens_slice(tokens);
        Murmur3Key murmur3_key;
        murmur_hash3(nullptr, tokens_slice, murmur3_key.data);

        auto iter = cached_sparse_mask_.find(murmur3_key);
        if (iter != cached_sparse_mask_.end()) {
          // Collect indices locally first
          for (int32_t vocab_idx : iter->second) {
            local_token_indices.push_back(static_cast<int64_t>(token_idx));
            local_vocab_indices.push_back(static_cast<int64_t>(vocab_idx));
          }
        } else {
          mask[token_idx] = empty_place_holder_;
          LOG(ERROR) << "Failed to generate mask for " << tokens;
        }
      }
    }

    // Merge local results to global batch (thread-safe)
    if (!local_token_indices.empty()) {
      std::lock_guard<std::mutex> lock(batch_mutex);
      global_batch_token_indices.insert(global_batch_token_indices.end(),
                                        local_token_indices.begin(),
                                        local_token_indices.end());
      global_batch_vocab_indices.insert(global_batch_vocab_indices.end(),
                                        local_vocab_indices.begin(),
                                        local_vocab_indices.end());
    }
  };

  if (use_threadpool_for_beam_expansion_) {
    // 分段处理优化：每个线程处理多个mask
    const size_t batch_size =
        std::max(1UL, (total_nums + thread_num_ - 1) / thread_num_);
    const size_t num_batches = (total_nums + batch_size - 1) / batch_size;

    std::vector<std::shared_ptr<std::promise<void>>> promises;
    std::vector<std::future<void>> futures;
    promises.reserve(num_batches);
    futures.reserve(num_batches);

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      auto promise = std::make_shared<std::promise<void>>();
      futures.push_back(promise->get_future());
      promises.push_back(promise);

      size_t start_idx = batch_idx * batch_size;
      size_t end_idx = std::min(start_idx + batch_size, total_nums);

      extra_threadpool_->schedule(
          [update_mask, start_idx, end_idx, promise]() mutable {
            update_mask(start_idx, end_idx);
            promise->set_value();
          });
    }

    for (auto& future : futures) {
      future.get();
    }
  } else {
    update_mask(0, total_nums);
  }

  // Global batch tensor operation after all threads complete
  if (!global_batch_token_indices.empty()) {
    auto token_indices =
        torch::tensor(global_batch_token_indices, torch::kInt64);
    auto vocab_indices =
        torch::tensor(global_batch_vocab_indices, torch::kInt64);
    torch::TensorOptions device_options =
        torch::dtype(torch::kInt64).device(device_);
    token_indices = safe_to(token_indices, device_options, true);
    vocab_indices = safe_to(vocab_indices, device_options, true);
    mask.index_put_({token_indices, vocab_indices}, 0.0f);
    // auto indices = torch::stack({token_indices, vocab_indices}, 1);
    // return indices;
  }

  return mask;
}
}  // namespace xllm