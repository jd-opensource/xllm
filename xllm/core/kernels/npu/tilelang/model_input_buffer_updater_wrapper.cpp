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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_MODEL_INPUT_BUFFER_UPDATER_REGISTRY_INC
#error "XLLM_TL_MODEL_INPUT_BUFFER_UPDATER_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int32_t kCompileMaxTokens = 16384;
constexpr int32_t kCompileMaxBatch = 1024;
constexpr int32_t kCompileMaxBlockTableLen = 8192;
constexpr int32_t kCompileMaxHiddenSize = 16384;
constexpr int32_t kMropeComponents = 3;

#include XLLM_TL_MODEL_INPUT_BUFFER_UPDATER_REGISTRY_INC

int32_t checked_int32(int64_t value, const char* name) {
  CHECK_GE(value, 0) << "TileLang model_input_buffer_updater: " << name
                     << " must be >= 0";
  CHECK_LE(value, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang model_input_buffer_updater: " << name
      << " exceeds int32 range";
  return static_cast<int32_t>(value);
}

bool is_supported_embedding_dtype(c10::ScalarType dtype) {
  return dtype == c10::ScalarType::Half || dtype == c10::ScalarType::BFloat16 ||
         dtype == c10::ScalarType::Float;
}

void check_defined_npu_int32_vector(const torch::Tensor& tensor,
                                    const char* name) {
  CHECK(tensor.defined()) << "TileLang model_input_buffer_updater: " << name
                          << " must be defined";
  CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang model_input_buffer_updater: " << name << " must be on NPU";
  CHECK_EQ(tensor.dtype(), torch::kInt32)
      << "TileLang model_input_buffer_updater: " << name << " must be int32";
  CHECK_EQ(tensor.dim(), 1)
      << "TileLang model_input_buffer_updater: " << name << " must be 1D";
  CHECK_EQ(tensor.stride(0), 1)
      << "TileLang model_input_buffer_updater: " << name
      << " must be contiguous";
}

void check_defined_npu_int32_matrix(const torch::Tensor& tensor,
                                    const char* name) {
  CHECK(tensor.defined()) << "TileLang model_input_buffer_updater: " << name
                          << " must be defined";
  CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang model_input_buffer_updater: " << name << " must be on NPU";
  CHECK_EQ(tensor.dtype(), torch::kInt32)
      << "TileLang model_input_buffer_updater: " << name << " must be int32";
  CHECK_EQ(tensor.dim(), 2)
      << "TileLang model_input_buffer_updater: " << name << " must be 2D";
  CHECK_EQ(tensor.stride(1), 1)
      << "TileLang model_input_buffer_updater: " << name
      << " last-dim stride must be 1";
  CHECK_GT(tensor.stride(0), 0)
      << "TileLang model_input_buffer_updater: " << name
      << " row stride must be > 0";
}

void check_defined_npu_embedding_matrix(const torch::Tensor& tensor,
                                        const char* name) {
  CHECK(tensor.defined()) << "TileLang model_input_buffer_updater: " << name
                          << " must be defined";
  CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang model_input_buffer_updater: " << name << " must be on NPU";
  CHECK(is_supported_embedding_dtype(tensor.scalar_type()))
      << "TileLang model_input_buffer_updater: unsupported embedding dtype for "
      << name;
  CHECK_EQ(tensor.dim(), 2)
      << "TileLang model_input_buffer_updater: " << name << " must be 2D";
  CHECK_EQ(tensor.stride(1), 1)
      << "TileLang model_input_buffer_updater: " << name
      << " last-dim stride must be 1";
  CHECK_GT(tensor.stride(0), 0)
      << "TileLang model_input_buffer_updater: " << name
      << " row stride must be > 0";
}

void check_same_device(const torch::Tensor& reference,
                       const torch::Tensor& tensor,
                       const char* name) {
  if (!tensor.defined()) {
    return;
  }
  CHECK_EQ(tensor.device(), reference.device())
      << "TileLang model_input_buffer_updater: " << name
      << " must be on the same device as src_tokens";
}

uint8_t* tensor_data_ptr_or_null(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return nullptr;
  }
  return reinterpret_cast<uint8_t*>(const_cast<void*>(tensor.data_ptr()));
}

ModelInputBufferUpdaterSpecialization build_runtime_specialization(
    c10::ScalarType embedding_dtype,
    bool with_mrope,
    bool with_input_embedding,
    bool with_linear_state_indices,
    bool with_q_cu_seq_lens) {
  const TilelangDType tilelang_embedding_dtype =
      with_input_embedding ? to_tilelang_dtype(embedding_dtype)
                           : TilelangDType::kFloat32;
  return make_model_input_buffer_updater_specialization(
      ModelInputBufferUpdaterEmbeddingDType{tilelang_embedding_dtype},
      ModelInputBufferUpdaterWithMrope{with_mrope ? 1 : 0},
      ModelInputBufferUpdaterWithInputEmbedding{with_input_embedding ? 1 : 0},
      ModelInputBufferUpdaterWithLinearStateIndices{
          with_linear_state_indices ? 1 : 0},
      ModelInputBufferUpdaterWithQCuSeqLens{with_q_cu_seq_lens ? 1 : 0});
}

}  // namespace

bool has_model_input_buffer_updater_specialization(
    c10::ScalarType embedding_dtype,
    bool with_mrope,
    bool with_input_embedding,
    bool with_linear_state_indices,
    bool with_q_cu_seq_lens) {
  if (with_input_embedding && !is_supported_embedding_dtype(embedding_dtype)) {
    return false;
  }
  const auto specialization =
      build_runtime_specialization(embedding_dtype,
                                   with_mrope,
                                   with_input_embedding,
                                   with_linear_state_indices,
                                   with_q_cu_seq_lens);
  return find_model_input_buffer_updater_kernel_entry(specialization) !=
         nullptr;
}

void model_input_buffer_updater(ModelInputBufferUpdaterParams& params) {
  check_defined_npu_int32_vector(params.src_tokens, "src_tokens");
  check_defined_npu_int32_vector(params.src_new_cache_slots,
                                 "src_new_cache_slots");
  check_defined_npu_int32_vector(params.src_q_seq_lens, "src_q_seq_lens");
  check_defined_npu_int32_vector(params.src_kv_seq_lens, "src_kv_seq_lens");
  check_defined_npu_int32_matrix(params.src_block_tables, "src_block_tables");

  check_defined_npu_int32_vector(params.dst_tokens, "dst_tokens");
  check_defined_npu_int32_vector(params.dst_new_cache_slots,
                                 "dst_new_cache_slots");
  check_defined_npu_int32_vector(params.dst_q_seq_lens, "dst_q_seq_lens");
  check_defined_npu_int32_vector(params.dst_kv_seq_lens, "dst_kv_seq_lens");
  check_defined_npu_int32_matrix(params.dst_block_tables, "dst_block_tables");

  const bool with_q_cu_seq_lens =
      params.src_q_cu_seq_lens.defined() || params.dst_q_cu_seq_lens.defined();
  const bool with_linear_state_indices =
      params.src_linear_state_indices.defined() ||
      params.dst_linear_state_indices.defined();
  const bool with_input_embedding = params.src_input_embedding.defined() ||
                                    params.dst_input_embedding.defined();

  CHECK_EQ(params.src_q_cu_seq_lens.defined(),
           params.dst_q_cu_seq_lens.defined())
      << "TileLang model_input_buffer_updater: q_cu_seq_lens src/dst must both "
         "be defined or both be undefined";
  CHECK_EQ(params.src_linear_state_indices.defined(),
           params.dst_linear_state_indices.defined())
      << "TileLang model_input_buffer_updater: linear_state_indices src/dst "
         "must both be defined or both be undefined";
  CHECK_EQ(params.src_input_embedding.defined(),
           params.dst_input_embedding.defined())
      << "TileLang model_input_buffer_updater: input_embedding src/dst must "
         "both be defined or both be undefined";

  const bool with_mrope = params.src_positions.dim() == 2;
  if (with_mrope) {
    check_defined_npu_int32_matrix(params.src_positions, "src_positions");
    check_defined_npu_int32_matrix(params.dst_positions, "dst_positions");
    CHECK_EQ(params.src_positions.size(0), kMropeComponents)
        << "TileLang model_input_buffer_updater: src_positions mRoPE axis dim "
           "must be 3";
    CHECK_EQ(params.dst_positions.size(0), kMropeComponents)
        << "TileLang model_input_buffer_updater: dst_positions mRoPE axis dim "
           "must be 3";
  } else {
    check_defined_npu_int32_vector(params.src_positions, "src_positions");
    check_defined_npu_int32_vector(params.dst_positions, "dst_positions");
  }

  if (with_q_cu_seq_lens) {
    check_defined_npu_int32_vector(params.src_q_cu_seq_lens,
                                   "src_q_cu_seq_lens");
    check_defined_npu_int32_vector(params.dst_q_cu_seq_lens,
                                   "dst_q_cu_seq_lens");
  }
  if (with_linear_state_indices) {
    check_defined_npu_int32_vector(params.src_linear_state_indices,
                                   "src_linear_state_indices");
    check_defined_npu_int32_vector(params.dst_linear_state_indices,
                                   "dst_linear_state_indices");
  }
  if (with_input_embedding) {
    check_defined_npu_embedding_matrix(params.src_input_embedding,
                                       "src_input_embedding");
    check_defined_npu_embedding_matrix(params.dst_input_embedding,
                                       "dst_input_embedding");
    CHECK_EQ(params.src_input_embedding.scalar_type(),
             params.dst_input_embedding.scalar_type())
        << "TileLang model_input_buffer_updater: input_embedding dtype "
           "mismatch";
  }

  check_same_device(params.src_tokens, params.src_positions, "src_positions");
  check_same_device(
      params.src_tokens, params.src_new_cache_slots, "src_new_cache_slots");
  check_same_device(params.src_tokens, params.src_q_seq_lens, "src_q_seq_lens");
  check_same_device(
      params.src_tokens, params.src_kv_seq_lens, "src_kv_seq_lens");
  check_same_device(
      params.src_tokens, params.src_q_cu_seq_lens, "src_q_cu_seq_lens");
  check_same_device(params.src_tokens,
                    params.src_linear_state_indices,
                    "src_linear_state_indices");
  check_same_device(
      params.src_tokens, params.src_block_tables, "src_block_tables");
  check_same_device(
      params.src_tokens, params.src_input_embedding, "src_input_embedding");
  check_same_device(params.src_tokens, params.dst_tokens, "dst_tokens");
  check_same_device(params.src_tokens, params.dst_positions, "dst_positions");
  check_same_device(
      params.src_tokens, params.dst_new_cache_slots, "dst_new_cache_slots");
  check_same_device(params.src_tokens, params.dst_q_seq_lens, "dst_q_seq_lens");
  check_same_device(
      params.src_tokens, params.dst_kv_seq_lens, "dst_kv_seq_lens");
  check_same_device(
      params.src_tokens, params.dst_q_cu_seq_lens, "dst_q_cu_seq_lens");
  check_same_device(params.src_tokens,
                    params.dst_linear_state_indices,
                    "dst_linear_state_indices");
  check_same_device(
      params.src_tokens, params.dst_block_tables, "dst_block_tables");
  check_same_device(
      params.src_tokens, params.dst_input_embedding, "dst_input_embedding");

  const int32_t actual_num_tokens =
      checked_int32(params.src_tokens.size(0), "actual_num_tokens");
  const int32_t padded_num_tokens =
      checked_int32(params.padded_num_tokens, "padded_num_tokens");
  const int32_t actual_batch_size =
      checked_int32(params.src_q_seq_lens.size(0), "actual_batch_size");
  const int32_t actual_block_table_len =
      checked_int32(params.src_block_tables.size(1), "actual_block_table_len");
  const int32_t src_block_table_stride = checked_int32(
      params.src_block_tables.stride(0), "src_block_table_stride");
  const int32_t dst_block_table_stride = checked_int32(
      params.dst_block_tables.stride(0), "dst_block_table_stride");
  const int32_t src_positions_stride =
      with_mrope
          ? checked_int32(params.src_positions.size(1), "src_positions_stride")
          : 0;
  const int32_t dst_positions_stride =
      with_mrope
          ? checked_int32(params.dst_positions.size(1), "dst_positions_stride")
          : 0;
  const int32_t actual_hidden_size =
      with_input_embedding ? checked_int32(params.src_input_embedding.size(1),
                                           "actual_hidden_size")
                           : 0;
  const int32_t src_input_embedding_stride =
      with_input_embedding ? checked_int32(params.src_input_embedding.stride(0),
                                           "src_input_embedding_stride")
                           : 0;
  const int32_t dst_input_embedding_stride =
      with_input_embedding ? checked_int32(params.dst_input_embedding.stride(0),
                                           "dst_input_embedding_stride")
                           : 0;

  CHECK_LE(padded_num_tokens, kCompileMaxTokens)
      << "TileLang model_input_buffer_updater: padded_num_tokens exceeds "
         "compiled limit "
      << kCompileMaxTokens;
  CHECK_LE(actual_batch_size, kCompileMaxBatch)
      << "TileLang model_input_buffer_updater: actual_batch_size exceeds "
         "compiled limit "
      << kCompileMaxBatch;
  CHECK_LE(actual_block_table_len, kCompileMaxBlockTableLen)
      << "TileLang model_input_buffer_updater: actual_block_table_len exceeds "
         "compiled limit "
      << kCompileMaxBlockTableLen;
  CHECK_LE(src_block_table_stride, kCompileMaxBlockTableLen)
      << "TileLang model_input_buffer_updater: src_block_table_stride exceeds "
         "compiled limit "
      << kCompileMaxBlockTableLen;
  CHECK_LE(dst_block_table_stride, kCompileMaxBlockTableLen)
      << "TileLang model_input_buffer_updater: dst_block_table_stride exceeds "
         "compiled limit "
      << kCompileMaxBlockTableLen;
  if (with_mrope) {
    CHECK_LE(src_positions_stride, kCompileMaxTokens)
        << "TileLang model_input_buffer_updater: src_positions_stride exceeds "
           "compiled limit "
        << kCompileMaxTokens;
    CHECK_LE(dst_positions_stride, kCompileMaxTokens)
        << "TileLang model_input_buffer_updater: dst_positions_stride exceeds "
           "compiled limit "
        << kCompileMaxTokens;
  }
  if (with_input_embedding) {
    CHECK_LE(actual_hidden_size, kCompileMaxHiddenSize)
        << "TileLang model_input_buffer_updater: actual_hidden_size exceeds "
           "compiled limit "
        << kCompileMaxHiddenSize;
    CHECK_LE(src_input_embedding_stride, kCompileMaxHiddenSize)
        << "TileLang model_input_buffer_updater: src_input_embedding_stride "
           "exceeds compiled limit "
        << kCompileMaxHiddenSize;
    CHECK_LE(dst_input_embedding_stride, kCompileMaxHiddenSize)
        << "TileLang model_input_buffer_updater: dst_input_embedding_stride "
           "exceeds compiled limit "
        << kCompileMaxHiddenSize;
  }

  CHECK_LE(actual_num_tokens, padded_num_tokens)
      << "TileLang model_input_buffer_updater: actual_num_tokens must be <= "
         "padded_num_tokens";
  CHECK_EQ(params.src_new_cache_slots.size(0), actual_num_tokens)
      << "TileLang model_input_buffer_updater: src_new_cache_slots size "
         "mismatch";
  CHECK_GE(params.dst_tokens.size(0), padded_num_tokens)
      << "TileLang model_input_buffer_updater: dst_tokens capacity mismatch";
  CHECK_GE(params.dst_new_cache_slots.size(0), padded_num_tokens)
      << "TileLang model_input_buffer_updater: dst_new_cache_slots capacity "
         "mismatch";
  CHECK_EQ(params.src_kv_seq_lens.size(0), actual_batch_size)
      << "TileLang model_input_buffer_updater: src_kv_seq_lens size mismatch";
  CHECK_EQ(params.src_block_tables.size(0), actual_batch_size)
      << "TileLang model_input_buffer_updater: src_block_tables batch mismatch";
  CHECK_GE(params.dst_q_seq_lens.size(0), actual_batch_size)
      << "TileLang model_input_buffer_updater: dst_q_seq_lens capacity "
         "mismatch";
  CHECK_GE(params.dst_kv_seq_lens.size(0), actual_batch_size)
      << "TileLang model_input_buffer_updater: dst_kv_seq_lens capacity "
         "mismatch";
  CHECK_GE(params.dst_block_tables.size(0), actual_batch_size)
      << "TileLang model_input_buffer_updater: dst_block_tables batch capacity "
         "mismatch";
  CHECK_GE(params.dst_block_tables.size(1), actual_block_table_len)
      << "TileLang model_input_buffer_updater: dst_block_tables width capacity "
         "mismatch";

  if (with_mrope) {
    CHECK_EQ(params.src_positions.size(1), actual_num_tokens)
        << "TileLang model_input_buffer_updater: src_positions token dim "
           "mismatch";
    CHECK_GE(params.dst_positions.size(1), padded_num_tokens)
        << "TileLang model_input_buffer_updater: dst_positions token capacity "
           "mismatch";
  } else {
    CHECK_EQ(params.src_positions.size(0), actual_num_tokens)
        << "TileLang model_input_buffer_updater: src_positions size mismatch";
    CHECK_GE(params.dst_positions.size(0), padded_num_tokens)
        << "TileLang model_input_buffer_updater: dst_positions capacity "
           "mismatch";
  }

  if (with_q_cu_seq_lens) {
    CHECK_EQ(params.src_q_cu_seq_lens.size(0), actual_batch_size)
        << "TileLang model_input_buffer_updater: src_q_cu_seq_lens size "
           "mismatch";
    CHECK_GE(params.dst_q_cu_seq_lens.size(0), actual_batch_size)
        << "TileLang model_input_buffer_updater: dst_q_cu_seq_lens capacity "
           "mismatch";
  }
  if (with_linear_state_indices) {
    CHECK_EQ(params.src_linear_state_indices.size(0), actual_batch_size)
        << "TileLang model_input_buffer_updater: src_linear_state_indices size "
           "mismatch";
    CHECK_GE(params.dst_linear_state_indices.size(0), actual_batch_size)
        << "TileLang model_input_buffer_updater: dst_linear_state_indices "
           "capacity mismatch";
  }
  if (with_input_embedding) {
    CHECK_EQ(params.src_input_embedding.size(0), actual_num_tokens)
        << "TileLang model_input_buffer_updater: src_input_embedding row "
           "mismatch";
    CHECK_EQ(params.dst_input_embedding.size(1), actual_hidden_size)
        << "TileLang model_input_buffer_updater: dst_input_embedding hidden "
           "size mismatch";
    CHECK_GE(params.dst_input_embedding.size(0), padded_num_tokens)
        << "TileLang model_input_buffer_updater: dst_input_embedding capacity "
           "mismatch";
  }

  const auto specialization = build_runtime_specialization(
      with_input_embedding ? params.src_input_embedding.scalar_type()
                           : c10::ScalarType::Float,
      with_mrope,
      with_input_embedding,
      with_linear_state_indices,
      with_q_cu_seq_lens);
  const auto* entry =
      find_model_input_buffer_updater_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang model_input_buffer_updater: no compiled variant. Available "
         "variants: "
      << available_model_input_buffer_updater_variant_keys();

  if (padded_num_tokens == 0 && actual_batch_size == 0) {
    return;
  }

  const int32_t device_id = params.src_tokens.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(tensor_data_ptr_or_null(params.src_tokens),
            tensor_data_ptr_or_null(params.src_positions),
            tensor_data_ptr_or_null(params.src_new_cache_slots),
            tensor_data_ptr_or_null(params.src_q_seq_lens),
            tensor_data_ptr_or_null(params.src_kv_seq_lens),
            tensor_data_ptr_or_null(params.src_q_cu_seq_lens),
            tensor_data_ptr_or_null(params.src_linear_state_indices),
            tensor_data_ptr_or_null(params.src_block_tables),
            tensor_data_ptr_or_null(params.src_input_embedding),
            tensor_data_ptr_or_null(params.dst_tokens),
            tensor_data_ptr_or_null(params.dst_positions),
            tensor_data_ptr_or_null(params.dst_new_cache_slots),
            tensor_data_ptr_or_null(params.dst_q_seq_lens),
            tensor_data_ptr_or_null(params.dst_kv_seq_lens),
            tensor_data_ptr_or_null(params.dst_q_cu_seq_lens),
            tensor_data_ptr_or_null(params.dst_linear_state_indices),
            tensor_data_ptr_or_null(params.dst_block_tables),
            tensor_data_ptr_or_null(params.dst_input_embedding),
            actual_num_tokens,
            padded_num_tokens,
            actual_batch_size,
            src_block_table_stride,
            dst_block_table_stride,
            actual_block_table_len,
            src_positions_stride,
            dst_positions_stride,
            src_input_embedding_stride,
            dst_input_embedding_stride,
            actual_hidden_size,
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
