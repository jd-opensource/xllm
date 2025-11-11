#pragma once
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "util/hash_util.h"
#include "util/threadpool.h"

namespace xllm {

class ValidPathFilter final {
 public:
  ValidPathFilter(const std::string valid_path_filter_file,
                  const int32_t vocab_size,
                  torch::ScalarType dtype,
                  torch::Device device);
  ValidPathFilter(const std::vector<std::vector<int32_t>>& tokens_list,
                  const int32_t vocab_size,
                  torch::ScalarType dtype,
                  torch::Device device);

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // output: [num_tokens, vocab_size]
  torch::Tensor forward(const std::vector<std::vector<int32_t>>& tokens_list);

 private:
  void init_cached_mask(const std::vector<std::vector<int32_t>>& tokens_list,
                        const int32_t vocab_size);

  // prepare mask using cached sparse mask
  torch::Tensor forward_sparse_mask(
      const std::vector<std::vector<int32_t>>& tokens_list);

  // Sparse storage: map from key to indices of candidate tokens.
  std::unordered_map<Murmur3Key,
                     std::vector<int32_t>,
                     FixedStringKeyHash,
                     FixedStringKeyEqual>
      cached_sparse_mask_;

  torch::Tensor empty_place_holder_;
  torch::Tensor first_token_mask_;

  bool init_cached_tokens_ = false;

  static float pre_mask_factor_;

  int32_t vocab_size_;

  torch::ScalarType dtype_ = torch::ScalarType::Undefined;

  torch::Device device_;

  int32_t thread_num_;
  std::unique_ptr<ThreadPool> extra_threadpool_;
  // 控制是否使用线程池进行beam expansion
  bool use_threadpool_for_beam_expansion_ = true;
};

}  // namespace xllm
