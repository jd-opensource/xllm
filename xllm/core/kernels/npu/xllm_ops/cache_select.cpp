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

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnn_select_unshared_kv.h"
#include "cache_select.h"

#define CHECK_ACL_SUCCESS(expr, msg) \
  do {                               \
    auto _ret = (expr);              \
    if (_ret != ACL_SUCCESS) {       \
      LOG(ERROR) << msg;             \
      throw std::runtime_error(msg); \
    }                                \
  } while (0)
namespace xllm_ops {

void cache_select(const torch::Tensor& beam_index,
                  std::vector<torch::Tensor> x_key_block,
                  std::vector<torch::Tensor> x_value_block,
                  const torch::Tensor& block_table,
                  const torch::Tensor& group_offset,
                  int64_t decode_step,
                  int64_t beam_size,
                  int64_t layer_num) {
  xllm_ops_utils::check_tensor(beam_index, "beam_index", "cache_select");
  xllm_ops_utils::check_tensor(block_table, "block_table", "cache_select");
  xllm_ops_utils::check_tensor(group_offset, "group_offset", "cache_select");
  // Validate tensor lists
  for (const auto& t : x_key_block) {
    xllm_ops_utils::check_tensor(t, "x_key_block[i]", "cache_select");
  }
  for (const auto& t : x_value_block) {
    xllm_ops_utils::check_tensor(t, "x_value_block[i]", "cache_select");
  }
  aclTensor* beam_index_ids = nullptr;
  aclTensor* group_offset_ids = nullptr;
  aclTensor* block_table_ids = nullptr;
  aclTensorList* x_key_block_list_ids = nullptr;
  aclTensorList* x_value_block_list_ids = nullptr;
  std::vector<aclTensor*> x_key_block_list_ids_vec;
  std::vector<aclTensor*> x_value_block_list_ids_vec;
  for (auto& x_key_block_tensor : x_key_block) {
    aclTensor* x_key_block_id = nullptr;
    xllm_ops_utils::create_acltensor(&x_key_block_id, x_key_block_tensor);
    x_key_block_list_ids_vec.push_back(x_key_block_id);
  }
  for (auto& x_value_block_tensor : x_value_block) {
    aclTensor* x_value_block_id = nullptr;
    xllm_ops_utils::create_acltensor(&x_value_block_id, x_value_block_tensor);
    x_value_block_list_ids_vec.push_back(x_value_block_id);
  }
  x_key_block_list_ids = aclCreateTensorList(x_key_block_list_ids_vec.data(),
                                             x_key_block_list_ids_vec.size());
  x_value_block_list_ids = aclCreateTensorList(
      x_value_block_list_ids_vec.data(), x_value_block_list_ids_vec.size());
  xllm_ops_utils::create_acltensor(&beam_index_ids, beam_index);
  xllm_ops_utils::create_acltensor(&group_offset_ids, group_offset);
  xllm_ops_utils::create_acltensor(&block_table_ids, block_table);
  int32_t device_id = beam_index.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  CHECK_ACL_SUCCESS(
      aclnnSelectUnsharedKVGetWorkspaceSize(beam_index_ids,
                                            block_table_ids,
                                            x_key_block_list_ids,
                                            x_value_block_list_ids,
                                            group_offset_ids,
                                            decode_step,
                                            beam_size,
                                            layer_num,
                                            x_key_block_list_ids,
                                            x_value_block_list_ids,
                                            &workspace_size,
                                            &executor),
      "cache_select: failed to get workspace size");
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "cache_select: failed to allocate workspace");
  }
  CHECK_ACL_SUCCESS(
      aclnnSelectUnsharedKV(workspace_addr, workspace_size, executor, stream),
      "cache_select: failed to perform cache select");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "cache_select: failed to synchronize stream");
  aclDestroyTensor(beam_index_ids);
  aclDestroyTensor(group_offset_ids);
  aclDestroyTensor(block_table_ids);
  aclDestroyTensorList(x_key_block_list_ids);
  aclDestroyTensorList(x_value_block_list_ids);
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "cache_select: failed to free workspace");
  }
}
}  // namespace xllm_ops