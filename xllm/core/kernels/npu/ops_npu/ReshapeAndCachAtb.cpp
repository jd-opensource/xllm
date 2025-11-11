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

using namespace std;
namespace atb {
using ReshapeAndCacheParam = atb::infer::ReshapeAndCacheParam;
void _npu_reshape_and_cache(const at::Tensor& key,
                            const at::Tensor& value,
                            at::Tensor& key_cache,
                            at::Tensor& value_cache,
                            const at::Tensor& slot_indices) {
  const c10::OptionalDeviceGuard device_guard(device_of(key));
  OpParamCache<ReshapeAndCacheParam>& reshapeAndCacheParamCache =
      OpParamCache<ReshapeAndCacheParam>::getInstance();
  ReshapeAndCacheParam reshapeparam;
  reshapeparam.compressType = ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED;

  auto key_cache_format = at_npu::native::get_npu_format(key_cache);
  auto value_cache_format = at_npu::native::get_npu_format(value_cache);
  bool is_key_cache_nz = (key_cache_format == ACL_FORMAT_FRACTAL_NZ);
  bool is_value_cache_nz = (value_cache_format == ACL_FORMAT_FRACTAL_NZ);

  if (is_key_cache_nz && is_value_cache_nz) {
    reshapeparam.kvCacheCfg = ReshapeAndCacheParam::K_CACHE_V_CACHE_NZ;
  } else {
    reshapeparam.kvCacheCfg = ReshapeAndCacheParam::K_CACHE_V_CACHE;
  }

  ParamSetter parametter;
  parametter.Input(key, true)
      .Input(value, true)
      .Input(key_cache)
      .Input(value_cache)
      .Input(slot_indices, true)
      .Output(key_cache)
      .Output(value_cache);
  auto opReshape = reshapeAndCacheParamCache.getOperation(
      reshapeparam, "ReshapeCacheOperation");
  RunAtbCmd(opReshape, parametter, "ReshapeCacheOperation");

  return;
}

}  // namespace atb
