// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <acl/acl.h>

#include "../custom_functions_npu/AtbCommon.h"

namespace atb {
using PagedAttentionParam = atb::infer::PagedAttentionParam;
void _npu_paged_attention(const at::Tensor& query,
                          const at::Tensor& key_cache,
                          const at::Tensor& value_cache,
                          int64_t num_kv_heads,
                          int64_t num_heads,
                          double scale_value,
                          const at::Tensor& block_table,
                          const at::Tensor& context_lens,
                          at::Tensor& out) {
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  OpParamCache<PagedAttentionParam>& pagedAttentionParamCache =
      OpParamCache<PagedAttentionParam>::getInstance();
  PagedAttentionParam pagedparam;
  pagedparam.headNum = num_heads;
  pagedparam.qkScale = scale_value;
  pagedparam.kvHeadNum = num_kv_heads;
  pagedparam.maskType = PagedAttentionParam::UNDEFINED;
  pagedparam.batchRunStatusEnable = false;
  pagedparam.quantType = PagedAttentionParam::TYPE_QUANT_UNDEFINED;
  pagedparam.outDataType = ACL_DT_UNDEFINED;
  pagedparam.hasQuantOffset = false;
  pagedparam.compressType = PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
  pagedparam.calcType = PagedAttentionParam::CALC_TYPE_UNDEFINED;
  pagedparam.scaleType = PagedAttentionParam::SCALE_TYPE_TOR;
  pagedparam.inputLayout = atb::infer::TYPE_BSND;
  pagedparam.mlaVHeadSize = 0;

  ParamSetter paramsetter;
  paramsetter.Input(query, true)
      .Input(key_cache)
      .Input(value_cache)
      .Input(block_table, true)
      .Input(context_lens, true)
      .Output(out);
  auto opPaged = pagedAttentionParamCache.getOperation(
      pagedparam, "PagedAttentionOperation");
  RunAtbCmd(opPaged, paramsetter, "PagedAttentionOperation");

  return;
}

}  // namespace atb