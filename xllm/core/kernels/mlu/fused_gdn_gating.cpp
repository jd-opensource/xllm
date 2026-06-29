/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "tmo_kernel/kernel/include/fused_gdn_gating.h"

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <glog/logging.h>

#include <unordered_map>

#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/mlu/utils.h"

namespace xllm {
namespace kernel {
namespace mlu {

std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float beta,
    float threshold) {
  int32_t batch = static_cast<int32_t>(a.size(0));
  int32_t num_heads = static_cast<int32_t>(a.size(1));
  int32_t seq_len = 1;

  // Create output tensors
  // g: (1, batch, num_heads) fp32
  torch::Tensor g =
      torch::empty({1, batch, num_heads}, a.options().dtype(torch::kFloat32));
  // beta_output: (1, batch, num_heads) same dtype as b (bf16)
  torch::Tensor beta_output =
      torch::empty({1, batch, num_heads}, b.options().dtype(b.dtype()));

  auto device_prop =
      torch_mlu::getDeviceProperties(torch_mlu::current_device());
  int32_t core_count = static_cast<int32_t>(device_prop->cluster_count *
                                            device_prop->core_num_per_cluster);

  constexpr int32_t kBlkHeads = 8;
  int32_t num_head_blocks = (num_heads + kBlkHeads - 1) / kBlkHeads;

  // Grid: (core_count, seq_len, num_head_blocks) — matches original
  //   grid = (TOTAL_CORE_NUM, seq_len, triton.cdiv(num_heads, 8))
  cnrtDim3_t dim_block = {static_cast<uint32_t>(core_count),
                          static_cast<uint32_t>(seq_len),
                          static_cast<uint32_t>(num_head_blocks)};

  auto queue = torch_mlu::getCurMLUStream();

  // algo_id: select pre-compiled kernel variant based on (NUM_HEADS,
  // core_count) Layout: for each core_num, iterate over all NUM_HEADS values.
  //   core_num=32  → offset 0:  (4,0), (8,1), (12,2), (16,3), (24,4), (32,5),
  //   (48,6), (64,7) core_num=64 → offset 8:  (4,8), (8,9), (12,10), (16,11),
  //   (24,12), (32,13), (48,14), (64,15)
  static const std::unordered_map<int32_t, int32_t> kNumHeadsToIdx = {
      {4, 0},
      {8, 1},
      {12, 2},
      {16, 3},
      {24, 4},
      {32, 5},
      {48, 6},
      {64, 7},
  };
  static const std::unordered_map<int32_t, int32_t> kCoreNumToOffset = {
      {32, 0},
      {64, 8},
  };
  int32_t algo_id =
      lookup_algo_id(kCoreNumToOffset, core_count, /*dim_name=*/"core_count") +
      lookup_algo_id(kNumHeadsToIdx, num_heads, /*dim_name=*/"num_heads");

  // Signature: *fp32, *bf16, *bf16, *bf16, *bf16, *bf16, i32, i32,
  //            {NUM_HEADS}, 1.0, 20.0, 8, {core_num}
  // Runtime args passed: pointers + seq_len + batch
  // Constexpr args (baked via algo_id): NUM_HEADS, beta, threshold, BLK_HEADS,
  // core_num
  tmo_fused_gdn_gating_kernel(queue,
                              &dim_block,
                              g.data_ptr(),
                              beta_output.data_ptr(),
                              A_log.data_ptr(),
                              a.data_ptr(),
                              b.data_ptr(),
                              dt_bias.data_ptr(),
                              seq_len,
                              batch,
                              algo_id);

  return std::make_pair(g, beta_output);
}

}  // namespace mlu
}  // namespace kernel
}  // namespace xllm
