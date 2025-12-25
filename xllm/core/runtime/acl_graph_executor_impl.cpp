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

#include "acl_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <numeric>

#include "core/common/global_flags.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/util/utils.h"

// ATB includes
#include <atb/atb_infer.h>
#include <atb/context.h>
#include <atb/operation.h>
#include <customize/custom_paged_attention_function.h>
#include <customize/customize_op_params.h>

#include "pytorch/adapter/utils/utils.h"

namespace xllm {

// GraphPersistentParam implementation
GraphPersistentParam::GraphPersistentParam(const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options,
                                           bool need_update_attn_mask)
    : args_(args),
      device_(device),
      options_(options),
      context_for_plan_(nullptr),
      custom_pa_op_for_plan_(nullptr),
      stream_for_plan_(nullptr),
      need_update_attn_mask_(need_update_attn_mask) {
  // Use max_tokens_per_batch for first dimension size
  // num_decode_tokens
  const int64_t max_tokens_per_batch = FLAGS_max_tokens_per_batch;
  // num_sequences
  const int64_t max_seqs_per_batch = options.max_seqs_per_batch();
  auto tensor_options = torch::TensorOptions().device(device);

  const int64_t max_seq_len = FLAGS_max_seq_len_for_graph_mode > 0
                                  ? FLAGS_max_seq_len_for_graph_mode
                                  : args_.max_position_embeddings();

  // Create persistent tensors with max_tokens_per_batch as first dimension
  persistent_tokens_ = torch::zeros({max_tokens_per_batch},
                                    torch::dtype(torch::kInt).device(device));
  persistent_positions_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_new_cache_slots_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));

  // Sequence length tensors with max_seqs_per_batch
  q_seq_lens_ = torch::zeros({max_seqs_per_batch},
                             torch::dtype(torch::kInt).device(device));
  kv_seq_lens_ = torch::zeros({max_seqs_per_batch},
                              torch::dtype(torch::kInt).device(device));

  // Block table tensors with maximum possible size
  const auto block_size = options.block_size();
  const int64_t max_block_table_len =
      (max_seq_len + block_size - 1) / block_size + 1;
  persistent_block_tables_ =
      torch::zeros({max_seqs_per_batch, max_block_table_len},
                   torch::dtype(torch::kInt).device(device));

  // Output tensor for hidden states
  torch::Dtype dtype = util::parse_dtype(args.dtype(), device);
  if (args.dtype() == "float" || args.dtype() == "float32") {
    LOG(WARNING)
        << "Acl graph executor init hidden_states compatible with float32 "
           "dtype: float32. This should not happen in production but for test.";
    dtype = torch::kFloat32;
  }
  hidden_states_ = torch::zeros({max_tokens_per_batch, args.hidden_size()},
                                torch::dtype(dtype).device(device));

  // Initialize persistent_mask_ if need_update_attn_mask is true
  if (need_update_attn_mask_) {
    persistent_mask_ = torch::zeros({max_tokens_per_batch, max_seq_len},
                                    torch::dtype(dtype).device(device));
  }

  // Do not need to create ATB context and custom paged attention operation
  if (args_.head_dim() == 0) {
    return;
  }

  initialize_paged_attention_plan_context(device);
}

GraphPersistentParam::~GraphPersistentParam() {
  if (custom_pa_op_for_plan_ != nullptr) {
    atb::DestroyOperation(custom_pa_op_for_plan_);
    custom_pa_op_for_plan_ = nullptr;
  }
  if (stream_for_plan_ != nullptr) {
    aclrtDestroyStream(stream_for_plan_);
    stream_for_plan_ = nullptr;
  }
  if (context_for_plan_ != nullptr) {
    atb::DestroyContext(context_for_plan_);
    context_for_plan_ = nullptr;
  }
}

void GraphPersistentParam::update(const torch::Tensor& tokens,
                                  const torch::Tensor& k_cache,
                                  const torch::Tensor& v_cache,
                                  const torch::Tensor& positions,
                                  const ModelInputParams& params,
                                  uint32_t actual_num_tokens) {
  // Copy data from input parameters to persistent graph tensors
  persistent_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(tokens, /*non_blocking=*/true);
  persistent_positions_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(positions, /*non_blocking=*/true);
  q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(params.q_seq_lens, /*non_blocking=*/true);
  kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(params.kv_seq_lens, /*non_blocking=*/true);
  persistent_new_cache_slots_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(params.new_cache_slots, /*non_blocking=*/true);

  // Copy block table data
  const int64_t actual_block_table_len = params.block_tables.size(1);
  const int64_t actual_batch_size = params.num_sequences;
  auto slice_persistent_block_tables =
      persistent_block_tables_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_block_table_len);
  slice_persistent_block_tables.copy_(params.block_tables,
                                      /*non_blocking=*/true);

  // Update persistent embedding from input_embedding if available
  const auto& embedding = params.input_embedding;
  if (embedding.defined()) {
    const int64_t embedding_tokens = embedding.size(0);

    // Initialize persistent_embedding_ if needed and not already initialized
    if (persistent_embedding_.numel() == 0) {
      const int64_t max_tokens_per_batch = FLAGS_max_tokens_per_batch;
      const int64_t embedding_dim = embedding.size(1);
      torch::Dtype dtype = util::parse_dtype(args_.dtype(), device_);
      persistent_embedding_ =
          torch::zeros({max_tokens_per_batch, embedding_dim},
                       torch::dtype(dtype).device(device_));
    }

    // Copy embedding data to persistent buffer
    persistent_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
        .copy_(embedding, /*non_blocking=*/true);
  }

  // Update attention mask only if needed
  if (need_update_attn_mask_) {
    update_attention_mask(params);
  }

  if (tiling_data_.numel() > 0) {
    // Get current stream for tiling tensor update
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    // Update tiling tensor
    plan_paged_attention_tiling(
        tokens, k_cache, v_cache, persistent_block_tables_, params, stream);
  }
}

void GraphPersistentParam::initialize_paged_attention_plan_context(
    const torch::Device& device) {
  // max paged attention tiling buffer size is 1024 * 256
  constexpr int64_t tiling_buffer_size = 1024 * 256;
  tiling_data_ = torch::zeros({tiling_buffer_size},
                              torch::dtype(torch::kInt32).device(device));

  // Initialize ATB context for paged attention plan
  atb::Status status = atb::customize::CreatePlanContext(&context_for_plan_);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to create ATB context for paged attention plan";

  // Create stream for paged attention plan
  aclError acl_status = aclrtCreateStream(&stream_for_plan_);
  CHECK_EQ(acl_status, ACL_SUCCESS)
      << "Failed to create ACL stream for paged attention plan";
  context_for_plan_->SetExecuteStream(stream_for_plan_);

  // Set launch mode to GRAPH_LAUNCH_MODE
  status = context_for_plan_->SetLaunchMode(atb::LaunchMode::GRAPH_LAUNCH_MODE);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to set launch mode to GRAPH_LAUNCH_MODE";

  // Create custom paged attention operation
  const int dp_local_tp_size = options_.world_size() / options_.dp_size();

  // Cache headNum and head_dim as member variables
  num_head_ = static_cast<int32_t>(args_.n_heads() / dp_local_tp_size);
  head_dim_ = static_cast<int32_t>(args_.head_dim());

  atb::customize::CustomPagedAttentionParam paOpParam;
  // default mask type is UNDEFINED, which means no mask is needed
  if (need_update_attn_mask_) {
    paOpParam.maskType =
        atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
  }
  paOpParam.headNum = num_head_;

  std::optional<long int> optionalValue = args_.n_kv_heads();
  paOpParam.kvHeadNum =
      std::max(1,
               static_cast<int32_t>(optionalValue.value_or(args_.n_heads())) /
                   dp_local_tp_size);

  const float head_dim_float = static_cast<float>(head_dim_);
  paOpParam.qkScale = 1.0f / std::sqrt(head_dim_float);

  const bool isBF16 = args_.dtype() == "bfloat16";
  if (isBF16) {
    paOpParam.outDataType = ACL_BF16;
  } else {
    paOpParam.outDataType = ACL_FLOAT16;
  }

  status = atb::CreateOperation(paOpParam, &custom_pa_op_for_plan_);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to create custom paged attention operation";
  CHECK_NE(custom_pa_op_for_plan_, nullptr) << "custom_pa_op_for_plan_ is null";
}

constexpr uint32_t TILING_PARA_SIZE = 17;
constexpr uint32_t TILING_HEAD_SIZE = 44;

namespace {
void parse_pa_host_tiling_buffer(const uint32_t* hostTilingBuffer,
                                 uint64_t tilingBufferSize) {
  VLOG(50) << "hostTilingBuffer.tilingBuffer: " << (void*)hostTilingBuffer;
  VLOG(50) << "hostTilingBuffer.tilingBufferSize: " << tilingBufferSize;
  if (hostTilingBuffer == nullptr || tilingBufferSize == 0) {
    VLOG(50) << "Invalid host tiling buffer!";
    return;
  }

  uint32_t tilingParamSize = tilingBufferSize / sizeof(uint32_t);
  VLOG(50) << "Total tiling param elements: " << tilingParamSize;

  // Parse header fields (TILING_HEAD_SIZE = 44)
  VLOG(50) << "\n=== Tiling Header Fields ===";
  VLOG(50) << "TILING_BATCH(tiling_head[0]): " << hostTilingBuffer[0];
  VLOG(50) << "TILING_NUMHEADS(tiling_head[1]): " << hostTilingBuffer[1];
  VLOG(50) << "TILING_HEADDIM(tiling_head[2]): " << hostTilingBuffer[2];
  VLOG(50) << "TILING_NUMBLOKS(tiling_head[3]): " << hostTilingBuffer[3];
  VLOG(50) << "TILING_BLOCKSIZE(tiling_head[4]): " << hostTilingBuffer[4];
  VLOG(50) << "TILING_MAXBLOCKS(tiling_head[5]): " << hostTilingBuffer[5];
  VLOG(50) << "TILING_TOR(tiling_head[6]): " << hostTilingBuffer[6];
  VLOG(50) << "TILING_KVHEADS(tiling_head[7]): " << hostTilingBuffer[7];
  VLOG(50) << "TILING_FORMER_BATCH(tiling_head[8]): " << hostTilingBuffer[8];
  VLOG(50) << "TILING_FORMER_HEAD(tiling_head[9]): " << hostTilingBuffer[9];
  VLOG(50) << "TILING_TAIL_BATCH(tiling_head[10]): " << hostTilingBuffer[10];
  VLOG(50) << "TILING_TAIL_HEAD(tiling_head[11]): " << hostTilingBuffer[11];
  VLOG(50) << "TILING_HEADNUM_MOVE(tiling_head[12]): " << hostTilingBuffer[12];
  VLOG(50) << "TILING_MASK_MAX_LEN(tiling_head[13]): " << hostTilingBuffer[13];
  VLOG(50) << "TILING_BATCH_STRIDE(tiling_head[14]): " << hostTilingBuffer[14];
  VLOG(50) << "TILING_HEAD_STRIDE(tiling_head[15]): " << hostTilingBuffer[15];
  VLOG(50) << "TILING_KEY(tiling_head[16]): " << hostTilingBuffer[16];
  VLOG(50) << "TILING_HEADSIZE(tiling_head[17]): " << hostTilingBuffer[17];
  VLOG(50) << "TILING_PARASIZE(tiling_head[18]): " << hostTilingBuffer[18];
  VLOG(50) << "TILING_GROUPNUM(tiling_head[19]): " << hostTilingBuffer[19];
  VLOG(50) << "TILING_FORMER_GROUP_MOVE(tiling_head[20]): "
           << hostTilingBuffer[20];
  VLOG(50) << "TILING_TAIL_GROUP_MOVE(tiling_head[21]): "
           << hostTilingBuffer[21];
  VLOG(50) << "TILING_MAX_KVSEQLEN(tiling_head[22]): " << hostTilingBuffer[22];
  VLOG(50) << "TILING_KVSPLIT(tiling_head[23]): " << hostTilingBuffer[23];
  VLOG(50) << "TILING_KVCORENUM(tiling_head[24]): " << hostTilingBuffer[24];
  VLOG(50) << "TILING_BLOCKSIZE_CALC(tiling_head[25]): "
           << hostTilingBuffer[25];
  VLOG(50) << "TILING_TOTAL_BLOCK_NUM(tiling_head[26]): "
           << hostTilingBuffer[26];
  VLOG(50) << "TILING_PREFILL_BS(tiling_head[27]): " << hostTilingBuffer[27];
  VLOG(50) << "TILING_DECODER_BS(tiling_head[28]): " << hostTilingBuffer[28];
  VLOG(50) << "TILING_HEADDIM_V(tiling_head[29]): " << hostTilingBuffer[29];
  VLOG(50) << "TILING_MODCOEF(tiling_head[30]): " << hostTilingBuffer[30];
  VLOG(50) << "TILING_DIVCOEF(tiling_head[31]): " << hostTilingBuffer[31];
  VLOG(50) << "TILING_QHEADORIGINAL(tiling_head[32]): " << hostTilingBuffer[32];
  VLOG(50) << "TILING_COMPRESSHEAD(tiling_head[33]): " << hostTilingBuffer[33];
  VLOG(50) << "TILING_QUANTYPE(tiling_head[34]): " << hostTilingBuffer[34];
  VLOG(50) << "TILING_DATA_SHAPE_TYPE(tiling_head[35]): "
           << hostTilingBuffer[35];
  VLOG(50) << "TILING_SCALETYPE(tiling_head[36]): " << hostTilingBuffer[36];
  VLOG(50) << "TILING_MASK_TYPE_ND(tiling_head[37]): " << hostTilingBuffer[37];
  VLOG(50) << "TILING_HEADDIM_K_SPLIT(tiling_head[38]): "
           << hostTilingBuffer[38];
  VLOG(50) << "TILING_HEADDIM_V_SPLIT(tiling_head[39]): "
           << hostTilingBuffer[39];
  VLOG(50) << "TILING_HEADDIM_V_SPLIT_VECTOR_FORMER(tiling_head[40]): "
           << hostTilingBuffer[40];
  VLOG(50) << "TILING_HEADDIM_V_SPLIT_VECTOR_TAIL(tiling_head[41]): "
           << hostTilingBuffer[41];
  VLOG(50) << "TILING_MTP_HEAD_SPLIT_SIZE(tiling_head[42]): "
           << hostTilingBuffer[42];
  VLOG(50) << "TILING_MTP_HEAD_SPLIT_NUM(tiling_head[43]): "
           << hostTilingBuffer[43];

  // Parse batch parameters
  if (tilingParamSize > TILING_HEAD_SIZE) {
    uint32_t batchCount = hostTilingBuffer[0];
    VLOG(50) << "\n=== Batch Parameters ===";
    VLOG(50) << "Number of batches: " << batchCount;
    batchCount = std::min(batchCount, 20u);

    for (uint32_t batchIdx = 0; batchIdx < batchCount; ++batchIdx) {
      uint32_t offset = TILING_HEAD_SIZE + batchIdx * TILING_PARA_SIZE;
      if (offset + TILING_PARA_SIZE <= tilingParamSize) {
        VLOG(50) << "\n--- Batch " << batchIdx << " ---";
        VLOG(50) << "  qSeqLen(batch_tiling_param[0]): "
                 << hostTilingBuffer[offset + 0];
        VLOG(50) << "  kvSeqLen(batch_tiling_param[1]): "
                 << hostTilingBuffer[offset + 1];
        VLOG(50) << "  qSBlockTile(batch_tiling_param[2]): "
                 << hostTilingBuffer[offset + 2];
        VLOG(50) << "  blockSize(batch_tiling_param[3]): "
                 << hostTilingBuffer[offset + 3];
        VLOG(50) << "  addrQSeqOffset[high](batch_tiling_param[4]): "
                 << hostTilingBuffer[offset + 4];
        VLOG(50) << "  addrQSeqOffset[low](batch_tiling_param[5]): "
                 << hostTilingBuffer[offset + 5];
        VLOG(50) << "  addrOSeqOffset[high](batch_tiling_param[6]): "
                 << hostTilingBuffer[offset + 6];
        VLOG(50) << "  addrOSeqOffset[low](batch_tiling_param[7]): "
                 << hostTilingBuffer[offset + 7];
        VLOG(50) << "  seqIdx(batch_tiling_param[8]): "
                 << hostTilingBuffer[offset + 8];
        VLOG(50) << "  totalQBlkNum(batch_tiling_param[9]): "
                 << hostTilingBuffer[offset + 9];
        VLOG(50) << "  maskOffset[high](batch_tiling_param[10]): "
                 << hostTilingBuffer[offset + 10];
        VLOG(50) << "  addrLSeqOffset[high](batch_tiling_param[11]): "
                 << hostTilingBuffer[offset + 11];
        VLOG(50) << "  addrLSeqOffset[low](batch_tiling_param[12]): "
                 << hostTilingBuffer[offset + 12];
        VLOG(50) << "  maskOffset[low](batch_tiling_param[14]): "
                 << hostTilingBuffer[offset + 14];
        VLOG(50) << "  addrOFdSeqOffset[high](batch_tiling_param[15]): "
                 << hostTilingBuffer[offset + 15];
        VLOG(50) << "  addrOFdSeqOffset[low](batch_tiling_param[16]): "
                 << hostTilingBuffer[offset + 16];
      }
    }
  }

  VLOG(50) << "\n=== End of Tiling Buffer Parse ===";
}
}  // namespace

void GraphPersistentParam::plan_paged_attention_tiling(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const ModelInputParams& input_params,
    aclrtStream stream) {
  // Convert torch tensors to atb tensors
  atb::Tensor atb_k_cache = atb_speed::Utils::AtTensor2Tensor(k_cache);
  atb::Tensor atb_v_cache = atb_speed::Utils::AtTensor2Tensor(v_cache);
  atb::Tensor atb_block_tables =
      atb_speed::Utils::AtTensor2Tensor(block_tables);
  // Get context_lens from input_params.kv_seq_lens
  atb::Tensor atb_context_lens =
      atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
  atb_context_lens.hostData =
      const_cast<int32_t*>(input_params.kv_seq_lens_vec.data());
  atb::Tensor atb_tiling_data = atb_speed::Utils::AtTensor2Tensor(tiling_data_);

  atb_tiling_data.desc.dtype = ACL_UINT32;

  // Construct query atb tensor from tokens: shape [num_tokens, headNum,
  // head_dim] Get number of tokens from tokens tensor
  const int64_t num_tokens = tokens.size(0);

  // Create query atb tensor with only desc (no actual data needed)
  atb::Tensor atb_query;
  // TODO: support quant dtype
  atb_query.desc.dtype = (args_.dtype() == "bfloat16") ? ACL_BF16 : ACL_FLOAT16;
  atb_query.desc.format = ACL_FORMAT_ND;
  atb_query.desc.shape.dimNum = 3;
  atb_query.desc.shape.dims[0] = num_tokens;
  atb_query.desc.shape.dims[1] = num_head_;
  atb_query.desc.shape.dims[2] = head_dim_;
  atb_query.deviceData = atb_k_cache.deviceData;
  atb_query.hostData = nullptr;
  // Calculate dataSize: num_tokens * headNum * head_dim * sizeof(dtype)
  // ACL_FLOAT16 and ACL_BF16 both use 2 bytes per element
  const uint64_t element_size =
      (atb_query.desc.dtype == ACL_BF16 || atb_query.desc.dtype == ACL_FLOAT16)
          ? 2
          : 1;
  atb_query.dataSize = static_cast<uint64_t>(num_tokens) *
                       static_cast<uint64_t>(num_head_) *
                       static_cast<uint64_t>(head_dim_) * element_size;

  atb::VariantPack custom_variantPack;
  // Conditionally include mask based on need_update_attn_mask_
  if (need_update_attn_mask_) {
    atb::Tensor atb_mask = atb_speed::Utils::AtTensor2Tensor(persistent_mask_);
    custom_variantPack.inTensors = {atb_query,
                                    atb_k_cache,
                                    atb_v_cache,
                                    atb_block_tables,
                                    atb_context_lens,
                                    atb_mask,
                                    atb_tiling_data};
  } else {
    // Skip mask when not needed
    custom_variantPack.inTensors = {atb_query,
                                    atb_k_cache,
                                    atb_v_cache,
                                    atb_block_tables,
                                    atb_context_lens,
                                    atb_tiling_data};
  }
  custom_variantPack.outTensors.push_back(atb_query);

  uint64_t custom_workspace_size = 0;
  atb::Status status = custom_pa_op_for_plan_->Setup(
      custom_variantPack, custom_workspace_size, context_for_plan_);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to setup custom paged attention operation for tiling";

  atb::customize::TilingBufferInfo tiling_buffer_info =
      atb::customize::GetHostTilingBufferFromCustomPagedAttentionOperation(
          custom_pa_op_for_plan_);

  CHECK_NE(tiling_buffer_info.tilingBuffer, nullptr)
      << "Tiling buffer is null after setup";
  CHECK_GT(tiling_buffer_info.tilingBufferSize, 0)
      << "Tiling buffer size is zero";

  if (VLOG_IS_ON(50)) {
    parse_pa_host_tiling_buffer((uint32_t*)tiling_buffer_info.tilingBuffer,
                                tiling_buffer_info.tilingBufferSize);
  }
  aclError acl_status =
      aclrtMemcpyAsync(tiling_data_.data_ptr(),
                       tiling_data_.numel() * sizeof(uint32_t),
                       tiling_buffer_info.tilingBuffer,
                       tiling_buffer_info.tilingBufferSize,
                       ACL_MEMCPY_HOST_TO_DEVICE,
                       stream);
  CHECK_EQ(acl_status, ACL_SUCCESS) << "Failed to copy tiling buffer to device";
}

void GraphPersistentParam::update_attention_mask(
    const ModelInputParams& input_params) {
  torch::Dtype dtype = util::parse_dtype(args_.dtype(), device_);

  // update persistent_mask_ in-place
  const int64_t batch_size = input_params.kv_seq_lens.size(0);
  const int64_t max_seq_len = input_params.kv_max_seq_len > 0
                                  ? input_params.kv_max_seq_len
                                  : FLAGS_max_seq_len_for_graph_mode;

  // persistent_mask_ is already initialized in constructor
  // Check if size is sufficient
  CHECK_LE(max_seq_len, persistent_mask_.size(1))
      << "max_seq_len (" << max_seq_len << ") exceeds max_seq_len ("
      << persistent_mask_.size(1) << ")";

  // Check if q_max_seq_len > 1 (prefill mode, not decode mode)
  bool chunked_prefill = input_params.q_max_seq_len > 1;

  // Calculate num_tokens: in chunked mode, sum of all q_len; in decode mode,
  // batch_size
  int64_t num_tokens = batch_size;  // Default for decode mode
  if (chunked_prefill) {
    CHECK_EQ(input_params.q_seq_lens_vec.size(), batch_size)
        << "q_seq_lens_vec size (" << input_params.q_seq_lens_vec.size()
        << ") != batch_size (" << batch_size << ")";
    num_tokens =
        std::accumulate(input_params.q_seq_lens_vec.begin(),
                        input_params.q_seq_lens_vec.begin() + batch_size,
                        int64_t(0));
  }

  // Check if num_tokens is within bounds
  CHECK_LE(num_tokens, persistent_mask_.size(0))
      << "num_tokens (" << num_tokens << ") exceeds max_tokens_per_batch ("
      << persistent_mask_.size(0) << ")";

  // Get slice for actual num_tokens (compatible with both chunked and
  // non-chunked)
  auto mask_slice =
      persistent_mask_.slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/max_seq_len);

  // Zero out the slice first
  mask_slice.zero_();

  const float mask_value = (dtype == torch::kFloat16)
                               ? -std::numeric_limits<float>::infinity()
                               : -9984.0f;

  if (chunked_prefill) {
    // Generate mask considering both q_seq_lens and kv_seq_lens
    // For each sequence, generate mask with shape [q_len, kv_len]
    // mask_slice is [num_tokens, max_seq_len], where num_tokens = sum of all
    // q_len

    // Check if kv_seq_lens_vec is available
    CHECK_EQ(input_params.kv_seq_lens_vec.size(), batch_size)
        << "kv_seq_lens_vec size (" << input_params.kv_seq_lens_vec.size()
        << ") != batch_size (" << batch_size << ")";

    int64_t offset = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      const int32_t q_len = input_params.q_seq_lens_vec[i];
      const int32_t kv_len = input_params.kv_seq_lens_vec[i];

      // For chunked mode, slice out q_len rows for this sequence
      // mask_slice is [num_tokens, max_seq_len]
      // Get [q_len, kv_len] slice from mask_slice[offset:offset+q_len, :kv_len]
      auto seq_mask_slice =
          mask_slice.slice(/*dim=*/0, /*start=*/offset, /*end=*/offset + q_len)
              .slice(
                  /*dim=*/1, /*start=*/0, /*end=*/kv_len);  // [q_len, kv_len]

      // Zero out the slice first
      seq_mask_slice.zero_();

      // Generate mask for this sequence: [q_len, kv_len]
      // Use tril to generate lower triangular mask
      int diagonal = kv_len - q_len;
      auto options = torch::TensorOptions().dtype(torch::kBool).device(device_);
      auto bias = torch::tril(torch::ones({q_len, kv_len}, options), diagonal);
      bias = ~bias;  // Invert: True positions need to be masked

      // Fill mask values directly
      seq_mask_slice.masked_fill_(bias, mask_value);

      // Update offset for next sequence
      offset += q_len;
    }
  } else {
    // Original logic: only consider kv_seq_lens (decode mode, q_len = 1 for
    // all)
    auto positions = torch::arange(max_seq_len, torch::kInt32)
                         .to(device_)
                         .unsqueeze(0)
                         .expand({batch_size, max_seq_len});

    auto context_lens_expanded = input_params.kv_seq_lens.to(torch::kInt32)
                                     .unsqueeze(1)
                                     .expand({batch_size, max_seq_len});

    auto mask_condition = positions >= context_lens_expanded;
    mask_slice.masked_fill_(mask_condition, mask_value);
  }
}

bool AclGraph::capture(CausalLM* model,
                       const ModelArgs& args,
                       const runtime::Options& options,
                       const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       const ModelInputParams& params,
                       std::vector<KVCache>& kv_cache,
                       uint32_t bucket_num_tokens) {
  // Save bucket num_tokens for this graph instance
  num_tokens_ = bucket_num_tokens;

  // Get actual num_tokens from tokens tensor
  // const uint32_t actual_num_tokens = tokens.size(0);

  auto& tensor_options = model->options();

  torch::npu::synchronize();

  // Begin graph capture using NPUGraph mempool for temporary tensor management
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(tensor_options.device().index()).stream();

  // Update persistent parameters with input data before capture
  const torch::Tensor& k_cache = kv_cache[0].get_k_cache();
  const torch::Tensor& v_cache = kv_cache[0].get_v_cache();
  const uint32_t actual_num_tokens = tokens.size(0);
  persistent_param_.update(
      tokens, k_cache, v_cache, positions, params, actual_num_tokens);

  // Create ModelInputParams using persistent buffers
  ModelInputParams graph_params = params;
  graph_params.kv_seq_lens = persistent_param_.kv_seq_lens(num_tokens_);
  graph_params.q_seq_lens = persistent_param_.q_seq_lens(num_tokens_);
  CHECK_GE(num_tokens_, actual_num_tokens)
      << "num_tokens_ >= actual_num_tokens";
  graph_params.kv_seq_lens_vec.resize(num_tokens_);
  graph_params.q_seq_lens_vec.resize(num_tokens_);
  for (int i = actual_num_tokens; i < num_tokens_; i++) {
    graph_params.kv_seq_lens_vec[i] = 1;
    graph_params.q_seq_lens_vec[i] = 1;
  }
  graph_params.num_sequences = num_tokens_;
  graph_params.batch_forward_type = BatchForwardType::DECODE;

  graph_params.new_cache_slots =
      persistent_param_.persistent_new_cache_slots(num_tokens_);
  graph_params.block_tables =
      persistent_param_.persistent_block_tables(num_tokens_);
  // Only set attn_mask if need_update_attn_mask_ is true
  if (persistent_param_.need_update_attn_mask()) {
    graph_params.graph_buffer.attn_mask =
        persistent_param_.persistent_mask(num_tokens_);
  }
  graph_params.graph_buffer.tiling_data = persistent_param_.tiling_data();

  // Set persistent embedding if available and original input has embedding
  const auto& original_embedding = params.input_embedding;
  if (original_embedding.defined()) {
    torch::Tensor persistent_embedding =
        persistent_param_.persistent_embedding(num_tokens_);
    if (persistent_embedding.numel() > 0) {
      graph_params.input_embedding = persistent_embedding;
    }
  }

  // Synchronize stream to ensure all data is copied to graph persistent buffers
  aclrtSynchronizeStream(stream);

  // Use secondary stream for graph capture to avoid blocking main stream
  bool need_restore_stream = false;
  if (c10_npu::getCurrentNPUStream(tensor_options.device().index()) ==
      c10_npu::getDefaultNPUStream(tensor_options.device().index())) {
    auto secondary_stream =
        c10_npu::getStreamFromPool(true, tensor_options.device().index());
    c10_npu::setCurrentNPUStream(secondary_stream);
    need_restore_stream = true;
  }
  LOG(INFO) << "capture begin, bucket_num_tokens: " << bucket_num_tokens
            << " actual_num_tokens: " << actual_num_tokens << std::endl;
  graph_.capture_begin();

  // Execute forward pass - NPUGraph mempool manages temporary tensors
  auto forward_result =
      model->forward({persistent_param_.persistent_tokens(num_tokens_)},
                     {persistent_param_.persistent_positions(num_tokens_)},
                     kv_cache,
                     {graph_params});

  // Store result in persistent buffer owned by NPUGraph mempool
  persistent_param_.set_hidden_states(forward_result);
  graph_.capture_end();
  if (need_restore_stream) {
    c10_npu::setCurrentNPUStream(
        c10_npu::getDefaultNPUStream(tensor_options.device().index()));
  }

  // Synchronize and test replay to verify graph capture
  aclrtSynchronizeStream(stream);

  graph_.replay();

  // aclrtSynchronizeStream(stream);
  return true;
}

torch::Tensor AclGraph::replay(const torch::Tensor& tokens,
                               const torch::Tensor& positions,
                               std::vector<KVCache>& kv_cache,
                               const ModelInputParams& params) {
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_LE(actual_num_tokens, num_tokens_)
      << "num_tokens mismatch: expected <= " << num_tokens_ << ", got "
      << actual_num_tokens;

  // Update persistent parameters with new input data
  const torch::Tensor& k_cache = kv_cache[0].get_k_cache();
  const torch::Tensor& v_cache = kv_cache[0].get_v_cache();
  persistent_param_.update(
      tokens, k_cache, v_cache, positions, params, actual_num_tokens);

  // Replay captured graph - NPUGraph mempool reuses temporary tensors
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  graph_.replay();

  // this is necessary to ensure the graph replay is completed
  // aclError st = aclrtSynchronizeStream(stream);
  // CHECK_EQ(st, ACL_SUCCESS)
  // << "aclrtSynchronizeStream failed, error code: " << st;

  // Return only the actual num_tokens portion of hidden states
  return get_hidden_states(actual_num_tokens);
}

AclGraphExecutorImpl::AclGraphExecutorImpl(CausalLM* model,
                                           const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  // Create single persistent parameter object shared by all AclGraph instances
  persistent_param_ =
      std::make_unique<GraphPersistentParam>(args_, device_, options_);
}

ForwardInput AclGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);
}

// Main execution method with graph optimization for decode phase
// tokens: [num_decode_tokens]
// positions: [num_decode_tokens] token pos in the sequence
// returns: [num_decode_tokens, hidden_size]
torch::Tensor AclGraphExecutorImpl::run(const torch::Tensor& tokens,
                                        const torch::Tensor& positions,
                                        std::vector<KVCache>& kv_caches,
                                        const ModelInputParams& params) {
  // no mirco batch in decode phase
  const torch::Tensor& tokens_tensor = tokens;
  const torch::Tensor& positions_tensor = positions;
  const ModelInputParams& params_single = params;
  // Identify decode phase using q_max_seq_len for precise detection
  // Decode phase: all sequences have q_seq_len == 1 (generating one token at a
  // time) Prefill phase: sequences have q_seq_len > 1 (processing multiple
  // prompt tokens) We also check empty_kv_cache to ensure KV cache is not empty
  // (not first forward pass)
  const bool in_decoding_phase =
      (params_single.q_max_seq_len == 1) && !params_single.empty_kv_cache;
  VLOG(50) << "in_decoding_phase: " << in_decoding_phase
           << " q_max_seq_len: " << params_single.q_max_seq_len
           << " empty_kv_cache: " << params_single.empty_kv_cache
           << " n_layers: " << args_.n_layers();
  // If not in decode phase, use eager mode directly without acl graph
  if (!in_decoding_phase || args_.n_layers() == 1) {
    VLOG(50) << "AclGraphExecutorImpl::run() in eager mode";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Only use acl graph in decode phase for performance optimization
  // Get actual num_tokens from tokens shape
  const uint32_t n_tokens = tokens_tensor.size(/*dim=*/0);
  const uint32_t actual_batch_size = n_tokens / options_.num_decoding_tokens();
  const uint32_t bucket_num_tokens = get_bucket_num_tokens(n_tokens);

  // Check if conditions are suitable for graph execution (replay or capture)
  const auto max_seq_len = FLAGS_max_seq_len_for_graph_mode > 0
                               ? FLAGS_max_seq_len_for_graph_mode
                               : args_.max_position_embeddings();
  const bool seq_len_supported = params_single.kv_max_seq_len <= max_seq_len;

  // Combined condition for graph capture support
  // ACL graph executor only supports single tensor inputs (no micro-batching)
  const bool capture_supported = seq_len_supported;

  // Early return if conditions are not suitable for graph operations
  if (!capture_supported) {
    LOG(FATAL) << "Not suitable for graph operations.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Check if captured graph exists for this bucket num_tokens
  auto it = graphs_.find(bucket_num_tokens);
  if (it != graphs_.end()) {
    // Replay the existing graph
    VLOG(50) << "AclGraphExecutorImpl::run() in replay mode";
    return it->second->replay(
        tokens_tensor, positions_tensor, kv_caches, params_single);
  }

  // Graph doesn't exist for this bucket num_tokens, try to create it lazily
  auto graph = std::make_unique<AclGraph>(*persistent_param_);
  VLOG(50) << "AclGraphExecutorImpl::run() in capture mode";
  bool capture_success = graph->capture(model_,
                                        args_,
                                        options_,
                                        tokens_tensor,
                                        positions_tensor,
                                        params_single,
                                        kv_caches,
                                        bucket_num_tokens);

  if (capture_success) {
    LOG(INFO) << "Lazy capturing ACL graph for bucket num_tokens: "
              << bucket_num_tokens << " (actual num_tokens: " << n_tokens
              << ") done";

    // Save the graph for future reuse
    graphs_[bucket_num_tokens] = std::move(graph);

    // Return the output from capture (no need to replay since capture
    // already executed)
    return graphs_[bucket_num_tokens]->get_hidden_states(n_tokens);
  }

  // Fallback to eager mode if capture fails
  LOG(ERROR) << "Failed to capture ACL graph for bucket num_tokens: "
             << bucket_num_tokens;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

void AclGraph::print_graph_tensors() const {
  VLOG(50) << "graph persistent_tokens_: "
           << persistent_param_.persistent_tokens();
  VLOG(50) << "graph persistent_positions_: "
           << persistent_param_.persistent_positions();
  VLOG(50) << "graph persistent_new_cache_slots_: "
           << persistent_param_.persistent_new_cache_slots();
  VLOG(50) << "graph q_seq_lens_: " << persistent_param_.q_seq_lens();
  VLOG(50) << "graph kv_seq_lens_: " << persistent_param_.kv_seq_lens();
  VLOG(50) << "graph persistent_block_tables_: "
           << persistent_param_.persistent_block_tables();
  VLOG(50) << "graph hidden_states_: " << persistent_param_.hidden_states();
  // VLOG(50) << "graph persistent_mask_: " <<
  // persistent_param_.persistent_mask();
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t AclGraphExecutorImpl::get_bucket_num_tokens(
    uint32_t num_tokens) const {
  if (FLAGS_enable_graph_no_padding) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  } else if (num_tokens <= 2) {
    return 2;
  } else if (num_tokens <= 4) {
    return 4;
  } else if (num_tokens <= 8) {
    return 8;
  } else {
    // For num_tokens > 16, use multiples of 16
    return ((num_tokens + 15) / 16) * 16;
  }
}

}  // namespace xllm
