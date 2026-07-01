#if defined(XLLM_TORCH_MUSA)

#include "kernels/musa/block_copy_api.h"
#include "kernels/musa/fused_qknorm_rope_api.h"
#include "kernels/musa/xattention_ops_api.h"

#include "kernel_cuda_ns_open.inc"

void block_copy(torch::Tensor key_cache_ptrs,
                torch::Tensor value_cache_ptrs,
                torch::Tensor src_block_indices,
                torch::Tensor dst_block_indices,
                torch::Tensor cum_sum,
                int64_t numel_per_block,
                torch::ScalarType cache_dtype) {
  ::xllm::kernel::musa::block_copy(key_cache_ptrs,
                                   value_cache_ptrs,
                                   src_block_indices,
                                   dst_block_indices,
                                   cum_sum,
                                   numel_per_block,
                                   cache_dtype);
}

void cache_select(const torch::Tensor& beam_index,
                  std::vector<torch::Tensor>& unshared_k_cache,
                  std::vector<torch::Tensor>& unshared_v_cache,
                  const torch::Tensor& block_table,
                  int64_t decode_step,
                  int64_t beam_size,
                  int64_t layer_num) {
  ::xllm::kernel::musa::cache_select(beam_index,
                                     unshared_k_cache,
                                     unshared_v_cache,
                                     block_table,
                                     decode_step,
                                     beam_size,
                                     layer_num);
}

void fused_qk_norm_rope(torch::Tensor& qkv,
                        int64_t num_heads_q,
                        int64_t num_heads_k,
                        int64_t num_heads_v,
                        int64_t head_dim,
                        double eps,
                        const torch::Tensor& q_weight,
                        const torch::Tensor& k_weight,
                        const torch::Tensor& cos_sin_cache,
                        bool interleaved,
                        const torch::Tensor& position_ids) {
  ::xllm::kernel::musa::fused_qk_norm_rope(qkv,
                                           num_heads_q,
                                           num_heads_k,
                                           num_heads_v,
                                           head_dim,
                                           eps,
                                           q_weight,
                                           k_weight,
                                           cos_sin_cache,
                                           interleaved,
                                           position_ids);
}

void beam_search(torch::Tensor acc_logprob,
                 torch::Tensor in_sequence_group,
                 torch::Tensor top_tokens,
                 torch::Tensor top_logprobs,
                 torch::Tensor out_acc_logprob,
                 torch::Tensor out_token_ids,
                 torch::Tensor out_token_index,
                 torch::Tensor out_beam_count_prefix_sums,
                 torch::Tensor out_sequence_group,
                 uint32_t batch_size,
                 uint32_t num_return_sequences,
                 uint32_t current_step) {
  ::xllm::kernel::musa::beam_search(acc_logprob,
                                    in_sequence_group,
                                    top_tokens,
                                    top_logprobs,
                                    out_acc_logprob,
                                    out_token_ids,
                                    out_token_index,
                                    out_beam_count_prefix_sums,
                                    out_sequence_group,
                                    batch_size,
                                    num_return_sequences,
                                    current_step);
}

#include "kernel_cuda_ns_close.inc"
#endif
