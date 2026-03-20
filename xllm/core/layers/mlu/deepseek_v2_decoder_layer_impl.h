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

#pragma once

#include <torch/torch.h>

#include <optional>

#include "attention.h"
#include "deepseek_v2_attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/dp_utils.h"
#include "layers/common/rms_norm.h"
#include "layers/mlu/deepseek_v2_sparse_moe_block.h"
#include "layers/mlu/deepseek_v32_sp_context.h"

namespace xllm {
namespace layer {

class DeepseekV2DecoderLayerTestPeer;

class DeepseekV2DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit DeepseekV2DecoderLayerImpl(const ModelContext& context,
                                      int32_t layer_id);

  ~DeepseekV2DecoderLayerImpl() override = default;

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

  void set_sequence_parallel_context(
      const v32_sp::DeepseekV32SPContext* sp_ctx) {
    sequence_parallel_context_ = sp_ctx;
    // 8192 is a magic number chosen empirically to balance memory savings for
    // long sequence scenarios when Sequence Parallel is enabled. It helps
    // reduce memory consumption in those cases.
    sp_ffn_chunk_size_ = sp_ctx != nullptr ? 8192 : -1;
  }

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  enum class PostAttnMode {
    kReplicated,
    kPackedLocal,
  };

  struct PostAttnCarrier {
    torch::Tensor ffn_in;
    torch::Tensor skip_local;
    PaddingInfo pad_info;
    PostAttnMode mode = PostAttnMode::kReplicated;
  };

  PostAttnCarrier build_post_attn_carrier(
      torch::Tensor x,
      const torch::Tensor& residual,
      DeepseekV2AttentionImpl::PostAttnLayout attn_layout);
  PostAttnCarrier build_post_attn_local(torch::Tensor x,
                                        const torch::Tensor& residual);

  bool can_keep_local_output(const PostAttnCarrier& carrier,
                             ProcessGroup* pg) const;
  bool can_sp_chunk(const ModelInputParams& input_params) const;
  torch::Tensor comm_out(torch::Tensor x,
                         const PostAttnCarrier& carrier,
                         ProcessGroup* pg) const;
  torch::Tensor run_mlp(torch::Tensor x, const ModelInputParams& input_params);
  torch::Tensor restore_ffn_output(torch::Tensor x,
                                   const PostAttnCarrier& carrier);
  torch::Tensor reduce_out(torch::Tensor x, ProcessGroup* pg) const;

  friend class DeepseekV2DecoderLayerTestPeer;

  // parallel args
  ParallelArgs parallel_args_;
  bool is_moe_layer_;

  DeepseekV2Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  DeepseekV2SparseMoEBlock sparse_moe_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};
  const v32_sp::DeepseekV32SPContext* sequence_parallel_context_ = nullptr;
  int64_t sp_ffn_chunk_size_ = -1;
};

TORCH_MODULE(DeepseekV2DecoderLayer);
}  // namespace layer
}  // namespace xllm
