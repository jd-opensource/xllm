#include "npu_dp_ep_padding.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "common/global_flags.h"
#include "framework/mapping_npu.h"

namespace xllm {

MappingNPU::Options get_mapping_options() {
  MappingNPU::Options options;
  options.dp_size(2)
      .tp_size(8)
      .moe_tp_size(16)
      .moe_ep_size(1)
      .pp_size(1)
      .sp_size(1);
  return options;
}

TEST(DpEpPaddingTest, Build) {
  std::string rank_table_file;
  MappingNPU::Options options = get_mapping_options();
  MappingNPU mapping(rank_table_file, 16, 0, options);
  nlohmann::json data = mapping.to_json();
  torch::Tensor token_size_per_dp_group = torch::tensor({10, 10});
  DpEpPadding dp_ep_padding(token_size_per_dp_group,
                            8,
                            data,
                            torch::Device(torch::kCPU),
                            torch::Dtype(torch::kInt32),
                            true);
  DpEpPaddingData dp_ep_padding_data = dp_ep_padding.build();
  LOG(INFO) << "attn_padding_idx:" << dp_ep_padding_data.attn_padding_idx();
  LOG(INFO) << "attn_unpadding_idx:" << dp_ep_padding_data.attn_unpadding_idx();
  LOG(INFO) << "ffn_padding_idx:" << dp_ep_padding_data.ffn_padding_idx();
  LOG(INFO) << "ffn_unpadding_idx:" << dp_ep_padding_data.ffn_unpadding_idx();
  LOG(INFO) << "lm_head_skip_padding_token_indices:"
            << dp_ep_padding_data.lm_head_skip_padding_token_indices();
  LOG(INFO) << "gather_prenorm_idx:" << dp_ep_padding_data.gather_prenorm_idx();
  LOG(INFO) << "padding_idx:" << dp_ep_padding_data.padding_idx();
  LOG(INFO) << "un_padding_idx:" << dp_ep_padding_data.un_padding_idx();
  LOG(INFO) << "dynamic_ep_idx:" << dp_ep_padding_data.dynamic_ep_idx();
  LOG(INFO) << "moe_idx:" << dp_ep_padding_data.moe_idx();
}

}  // namespace xllm
