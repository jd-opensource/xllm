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

#include <acl/acl.h>

#include "../custom_functions_npu/atb_common.h"

using namespace std;
namespace atb {
using SelfAttentionParam = atb::infer::SelfAttentionParam;
void _npu_flash_attention(const at::Tensor& query,
                          const at::Tensor& key,
                          const at::Tensor& value,
                          const at::Tensor& mask,
                          const at::Tensor& seq_len,
                          const double scale_value,
                          const int64_t num_heads,
                          const int64_t num_kv_heads,
                          at::Tensor& out) {
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  OpParamCache<SelfAttentionParam>& selfAttentionParamCache =
      OpParamCache<SelfAttentionParam>::getInstance();
  SelfAttentionParam selfattentionparam;

  selfattentionparam.calcType = SelfAttentionParam::PA_ENCODER;
  selfattentionparam.kernelType = SelfAttentionParam::KERNELTYPE_DEFAULT;
  selfattentionparam.clampType = SelfAttentionParam::CLAMP_TYPE_UNDEFINED;
  selfattentionparam.maskType = SelfAttentionParam::MASK_TYPE_NORM;
  selfattentionparam.kvcacheCfg = SelfAttentionParam::K_CACHE_V_CACHE;
  selfattentionparam.scaleType = SelfAttentionParam::SCALE_TYPE_TOR;
  selfattentionparam.quantType = SelfAttentionParam::TYPE_QUANT_UNDEFINED;
  selfattentionparam.cacheType = SelfAttentionParam::CACHE_TYPE_NORM;
  selfattentionparam.outDataType = ACL_DT_UNDEFINED;
  selfattentionparam.headNum = num_heads;
  selfattentionparam.kvHeadNum = num_kv_heads;
  selfattentionparam.qScale = 1;
  selfattentionparam.qkScale = scale_value;
  selfattentionparam.batchRunStatusEnable = false;
  selfattentionparam.isTriuMask = 0;
  selfattentionparam.clampMin = 0;
  selfattentionparam.clampMax = 0;
  selfattentionparam.inputLayout = atb::infer::TYPE_BSND;
  selfattentionparam.mlaVHeadSize = 0;
  selfattentionparam.windowSize = 0;

  ParamSetter parametter;
  parametter.Input(query, true)
      .Input(key, true)
      .Input(value, true)
      .Input(mask)
      .Input(seq_len, true)
      .Output(out);

  auto opSelfattention = selfAttentionParamCache.get_operation(
      selfattentionparam, "SelfAttentionOperation");
  run_atb_cmd(opSelfattention, parametter, "SelfAttentionOperation");

  return;
}

}  // namespace atb