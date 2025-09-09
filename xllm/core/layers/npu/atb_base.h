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

#include <absl/strings/match.h>
#include <torch/torch.h>

#include <atomic>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "atb/atb_infer.h"
#include "buffer/atb_workspace.h"
#include "framework/context.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "pytorch/adapter/utils/utils.h"
#include "pytorch/adapter/workspace/workspace.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/singleton.h"
#include "xllm_kernels/core/include/atb_speed/utils/tensor_util.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
namespace xllm::hf {

enum class TransposeType : int {
  INVALID = -1,
  NOT_TRANSPOSE = 0,
  TRANSPOSE = 1
};

enum class LinearType : int { INVALID = -1, FP = 0, INT = 1 };

enum class PackType : int {
  PACK_QUANT_UNDEFINED = 0,
  ALL_FP = 1,
  ALL_W8A8 = 2,
  ALL_W8A8_ANTI = 3,
  MIX_W8A8 = 4,
  MIX_W8A8_ANTI = 5,
  ALL_W8A16 = 6,
  ALL_W8A8SC = 7,
  MIX_W8A8SC = 8,
  ALL_W8A8SC_ANTI = 9,
  MIX_W8A8SC_ANTI = 10,
  ALL_W4A16 = 11,
  ALL_W8A16_ANTI = 12,
  ALL_W4A16_ANTI = 13,
  MIX_W4A16 = 14,
  MIX_W4A16_ANTI = 15,
  MIX_W8A16 = 16,
  MIX_W8A16_ANTI = 17,
  ALL_W8A8_DYNAMIC = 18,
  ALL_W8A8_DYNAMIC_ANTI = 19,
  MIX_W8A8_DYNAMIC = 20,
  MIX_W8A8_DYNAMIC_ANTI = 21
};

enum class LinearTypeV2 : int {
  INVALID = -1,
  FLOAT16 = 0,
  BFLOAT16 = 1,
  W4A16 = 2,
  W8A16 = 3,
  W8A8 = 4,
  W8A8S = 5,
  W8A8SC = 6,
  W8A8_DYNAMIC = 7,
  W8A8_PDMIX = 8,
  W4A8_DYNAMIC = 9
};

class ATBBase {
 public:
  ATBBase(const Context& context);
  virtual ~ATBBase() {};

  using Task = std::function<int()>;
  using RunTaskFunc =
      std::function<void(const std::string& taskName, Task task)>;

  virtual void load_state_dict(const StateDict& state_dict) {};

  virtual void verify_loaded_weights() const {};

  virtual void merge_loaded_weights() {};

  virtual int64_t init_layer() { return 0; };

  void print_atbtensor(const atb::Tensor& tensor, int i);

  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position,
                  int dim);

  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position);

  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position,
                  int dim,
                  int rank,
                  int world_size);
  // void get_sharded(at::Tensor weight_tensor,int dim);

  atb::Status execute_node(atb_speed::Model::Node& node,
                           atb::Context* context,
                           AtbWorkspace& workspace,
                           int nodeId = 0,
                           aclrtEvent* event = nullptr,
                           std::atomic<bool>* event_flag = nullptr);

  atb::Status execute_plan(const atb_speed::Model::Node& node,
                           std::string opName_,
                           aclrtEvent* event,
                           std::atomic<bool>* event_flag);

  void run_task(std::string taskName, std::function<int()> task) const;

  atb::Tensor XTensor2Tensor(const std::shared_ptr<xllm::XTensor>& xtensor);

  torch::Dtype string_2_dtype(const std::string& dtype_str);

  void correct_tensor_dtype(torch::Tensor& tensor,
                            const std::string& tensorName);

 protected:
  atb::Context* context_;
  std::vector<at::Tensor> at_weight_tensors_;
  std::vector<atb::Tensor> atb_weight_tensors_;
  at::Device device_;
  std::string name_;
  torch::ScalarType dtype_;
  std::vector<int> placeholder_vec_;
  xllm::ParallelArgs parallel_args_;
  std::function<void(const std::string&, std::function<int()>)> runTaskFunc_;
  std::string quantize_type_;
  std::string torch_dtype_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  int32_t dp_rank_;
  int32_t dp_local_tp_rank_;
};

}  // namespace xllm::hf
