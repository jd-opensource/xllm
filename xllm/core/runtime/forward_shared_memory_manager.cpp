/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.
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

#include "forward_shared_memory_manager.h"

#include <gflags/gflags.h>

#include <cstring>
#include <stdexcept>

#include "core/common/global_flags.h"
#include "core/util/net.h"
#include "util/utils.h"

#define INLINE __attribute__((inline))

#if defined(__GNUC__)
static inline bool(likely)(bool x) { return __builtin_expect((x), true); }
static inline bool(unlikely)(bool x) { return __builtin_expect((x), false); }
#else
static inline bool(likely)(bool x) { return x; }
static inline bool(unlikely)(bool x) { return x; }
#endif

namespace xllm {

template <typename T>
constexpr size_t type_size = sizeof(T);

constexpr size_t sampling_param_fixed_size() {
  return 5 * type_size<float>      // frequency_penalty, presence_penalty,
                                   // repetition_penalty, temperature, top_p
         + 2 * type_size<int64_t>  // top_k, top_logprobs
         + 3 * type_size<bool>     // logprobs, do_sample, is_embeddings
         + type_size<int32_t>;     // beam_width
}

constexpr size_t swap_block_info_fixed_size() {
  return type_size<int32_t> * 2;  // src_block_id + dst_block_id
}

INLINE size_t get_string_size(const std::string& str) {
  return type_size<uint64_t> + str.size();
}

template <typename T>
INLINE size_t get_vector_size(const std::vector<T>& vec) {
  return type_size<uint64_t> + vec.size() * type_size<T>;
}

INLINE size_t get_tensor_size(const torch::Tensor& tensor) {
  uint64_t size = type_size<uint64_t>;             // ndim
  size += type_size<uint64_t> * tensor.dim();      // shape
  size += type_size<int8_t>;                       // dtype
  size += type_size<uint64_t>;                     // databytes
  size += tensor.numel() * tensor.element_size();  // data
  return size;
}

template <typename T>
INLINE size_t get_2d_vector_size(const std::vector<std::vector<T>>& vec2d) {
  size_t size = type_size<uint64_t>;
  for (const auto& vec : vec2d) {
    size += get_vector_size(vec);
  }
  return size;
}

INLINE size_t get_instance_info_size(const InstanceInfo& info) {
  size_t size = get_string_size(info.name) + get_string_size(info.rpc_address) +
                get_string_size(info.type);

  size += type_size<uint64_t> + info.cluster_ids.size() * type_size<uint64_t>;

  size += type_size<uint64_t>;
  for (const auto& addr : info.addrs) {
    size += get_string_size(addr);
  }

  size += type_size<uint64_t> + info.k_cache_ids.size() * type_size<int64_t> +
          type_size<uint64_t> + info.v_cache_ids.size() * type_size<int64_t> +
          type_size<int32_t>  // dp_size
          + type_size<uint64_t> +
          info.ttft_profiling_data.size() *
              (type_size<int32_t> + type_size<int64_t>);

  return size;
}

INLINE size_t get_transfer_kv_info_size(const TransferKVInfo& info) {
  return get_string_size(info.request_id) +
         get_vector_size(info.local_blocks_ids) +
         get_vector_size(info.remote_blocks_ids) +
         type_size<int32_t>  // dp_rank
         + get_instance_info_size(info.remote_instance_info);
}

INLINE size_t get_eplb_info_size(const EplbInfo& info) {
  return type_size<int32_t>  // prepare_layer_id
         + get_vector_size(info.expert_ids) +
         type_size<int32_t>;  // update_layer_id
}

INLINE size_t get_mm_batch_data_size(const MMBatchData& mm_data) {
  size_t total = 0;
  auto& data = mm_data.data();
  total += type_size<size_t> + type_size<uint32_t>;  // mm_dict size + mm_type
  for (auto& [mm_key, mm_value] : data) {
    total += get_string_size(mm_key);
    total += type_size<int32_t>;  // num of tensors
    if (std::holds_alternative<torch::Tensor>(mm_value)) {
      total += get_tensor_size(std::get<torch::Tensor>(mm_value));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(mm_value)) {
      for (const auto& tensor :
           std::get<std::vector<torch::Tensor>>(mm_value)) {
        total += get_tensor_size(tensor);
      }
    }
  }
  return total;
}

INLINE size_t calculate_raw_forward_input_size(const RawForwardInput& input) {
  size_t total = 0;

  const auto* vec1d = &input.flatten_tokens_vec;
  total += get_vector_size(*vec1d++);  // flatten_tokens_vec
  total += get_vector_size(*vec1d++);  // flatten_positions_vec
  total += get_vector_size(input.selected_token_idxes);
  total += get_vector_size(input.sample_idxes);
  total += get_vector_size(input.unique_token_lens_vec);
  total += get_vector_size(input.seq_lens);
  total += get_vector_size(input.q_seq_lens);
  total += get_vector_size(input.new_token_slot_ids);
  total += get_vector_size(input.dp_global_token_nums);
  total += get_vector_size(input.dp_is_decode);
  total += get_vector_size(input.embedding_ids);
  total += get_vector_size(input.src_block_indices);
  total += get_vector_size(input.dst_block_indices);
  total += get_vector_size(input.cum_sum);
  total += get_vector_size(input.new_cache_slot_offsets);
  total += get_vector_size(input.kv_cache_start_offsets);
  total += get_vector_size(input.acc_logprob_vec);
  total += get_vector_size(input.extra_token_ids);

  total += get_2d_vector_size(input.unique_token_ids_vec);
  total += get_2d_vector_size(input.unique_token_counts_vec);
  total += get_2d_vector_size(input.block_tables_vec);
  total += get_2d_vector_size(input.embeddings);

  total += type_size<uint64_t> +
           input.sampling_params.size() * sampling_param_fixed_size();

  total += type_size<uint64_t>;
  for (const auto& t : input.transfer_kv_infos) {
    total += get_transfer_kv_info_size(t);
  }

  total += type_size<uint64_t> +
           input.swap_blocks.size() * swap_block_info_fixed_size();

  total += type_size<bool> * 2        // empty_kv_cache + global_empty_kv_cache
           + type_size<int32_t>       // batch_forward_type
           + type_size<uint32_t> * 2  // max_seq_len + q_max_seq_len
           + type_size<int32_t>       // num_sequences
           + get_eplb_info_size(input.eplb_info);
  // m_position
  total += get_2d_vector_size(input.m_positions_vec);
  total += get_mm_batch_data_size(input.mm_data);

  return total;
}

template <typename T>
INLINE void write_data(char*& buffer, const T& data) {
  *reinterpret_cast<T*>(buffer) = data;
  buffer += type_size<T>;
}

INLINE void write_string(char*& buffer, const std::string& str) {
  const uint64_t len = str.size();
  write_data(buffer, len);
  if (len > 0) {
    std::memcpy(buffer, str.data(), len);
    buffer += len;
  }
}

INLINE void write_tensor(char*& buffer, const torch::Tensor& tensor) {
  auto contig_tensor = tensor.cpu().contiguous();
  // write dtype
  const int8_t tensor_dtype = static_cast<int8_t>(contig_tensor.scalar_type());
  write_data(buffer, tensor_dtype);
  // write ndim
  const uint64_t tensor_ndim = contig_tensor.dim();
  write_data(buffer, tensor_ndim);
  // write shape
  for (int64_t i = 0; i < contig_tensor.dim(); ++i) {
    write_data(buffer, static_cast<uint64_t>(contig_tensor.size(i)));
  }
  // write data_bytes
  const uint64_t tensor_data_bytes =
      contig_tensor.numel() * contig_tensor.element_size();
  write_data(buffer, tensor_data_bytes);

  if (tensor_data_bytes > 0) {
    std::memcpy(buffer, contig_tensor.data_ptr(), tensor_data_bytes);
    buffer += tensor_data_bytes;
  }
}

INLINE void write_sampling_param(char*& buffer,
                                 const RequestSamplingParam& param) {
  char* ptr = buffer;
  *reinterpret_cast<float*>(ptr) = param.frequency_penalty;
  ptr += type_size<float>;
  *reinterpret_cast<float*>(ptr) = param.presence_penalty;
  ptr += type_size<float>;
  *reinterpret_cast<float*>(ptr) = param.repetition_penalty;
  ptr += type_size<float>;
  *reinterpret_cast<float*>(ptr) = param.temperature;
  ptr += type_size<float>;
  *reinterpret_cast<float*>(ptr) = param.top_p;
  ptr += type_size<float>;
  *reinterpret_cast<int64_t*>(ptr) = param.top_k;
  ptr += type_size<int64_t>;
  *reinterpret_cast<bool*>(ptr) = param.logprobs;
  ptr += type_size<bool>;
  *reinterpret_cast<int64_t*>(ptr) = param.top_logprobs;
  ptr += type_size<int64_t>;
  *reinterpret_cast<bool*>(ptr) = param.do_sample;
  ptr += type_size<bool>;
  *reinterpret_cast<bool*>(ptr) = param.is_embeddings;
  ptr += type_size<bool>;
  *reinterpret_cast<int32_t*>(ptr) = param.beam_width;
  ptr += type_size<int32_t>;
  buffer = ptr;
}

template <typename T>
INLINE void write_vector(char*& buffer, const std::vector<T>& vec) {
  const uint64_t size = vec.size();
  write_data(buffer, size);
  if (size > 0) {
    const size_t bytes = size * type_size<T>;
    std::memcpy(buffer, vec.data(), bytes);
    buffer += bytes;
  }
}

template <typename T>
INLINE void write_2d_vector(char*& buffer,
                            const std::vector<std::vector<T>>& vec2d) {
  write_data(buffer, (uint64_t)vec2d.size());
  for (const auto& vec : vec2d) {
    write_vector(buffer, vec);
  }
}

INLINE void write_instance_info(char*& buffer, const InstanceInfo& info) {
  write_string(buffer, info.name);
  write_string(buffer, info.rpc_address);
  write_string(buffer, info.type);

  write_vector(buffer, info.cluster_ids);

  write_data(buffer, (uint64_t)info.addrs.size());
  for (const auto& addr : info.addrs) {
    write_string(buffer, addr);
  }

  write_vector(buffer, info.k_cache_ids);
  write_vector(buffer, info.v_cache_ids);
  write_data(buffer, info.dp_size);

  const uint64_t prof_size = info.ttft_profiling_data.size();
  write_data(buffer, prof_size);
  if (prof_size > 0) {
    std::memcpy(buffer,
                info.ttft_profiling_data.data(),
                prof_size * sizeof(std::pair<int32_t, int64_t>));
    buffer += prof_size * sizeof(std::pair<int32_t, int64_t>);
  }
}

INLINE void write_transfer_kv_info(char*& buffer, const TransferKVInfo& info) {
  write_string(buffer, info.request_id);
  write_vector(buffer, info.local_blocks_ids);
  write_vector(buffer, info.remote_blocks_ids);
  write_data(buffer, info.dp_rank);
  write_instance_info(buffer, info.remote_instance_info);
}

INLINE void write_eplb_info(char*& buffer, const EplbInfo& info) {
  write_data(buffer, info.prepare_layer_id);
  write_vector(buffer, info.expert_ids);
  write_data(buffer, info.update_layer_id);
}

INLINE void write_swap_blocks(char*& buffer,
                              const std::vector<BlockTransferInfo>& blocks) {
  write_data(buffer, (uint64_t)blocks.size());

  for (const auto& b : blocks) {
    *reinterpret_cast<int32_t*>(buffer) = b.src_block_id;
    *reinterpret_cast<int32_t*>(buffer + 4) = b.src_block_id;
    buffer += swap_block_info_fixed_size();
  }
}

INLINE void write_mm_batch_data(char*& buffer, const MMBatchData& mm_data) {
  auto& mm_dict = mm_data.data();
  // size
  size_t size = mm_dict.size();
  write_data(buffer, (size_t)size);
  // mm_type
  uint32_t mm_type = mm_data.type();
  write_data(buffer, mm_type);
  // tensor num
  int32_t tensor_num = 1;
  for (auto& [mm_key, mm_value] : mm_dict) {
    write_string(buffer, mm_key);

    if (std::holds_alternative<torch::Tensor>(mm_value)) {
      tensor_num = 1;
      write_data(buffer, tensor_num);
      auto& tensor = std::get<torch::Tensor>(mm_value);
      write_tensor(buffer, tensor);
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(mm_value)) {
      auto& tensor_vec = std::get<std::vector<torch::Tensor>>(mm_value);
      tensor_num = tensor_vec.size();
      write_data(buffer, tensor_num);
      for (const auto& tensor : tensor_vec) {
        write_tensor(buffer, tensor);
      }
    }
  }
}

template <typename T>
INLINE void read_data(const char*& buffer, T& data) {
  data = *reinterpret_cast<const T*>(buffer);
  buffer += type_size<T>;
}

INLINE void read_string(const char*& buffer, std::string& str) {
  uint64_t len;
  read_data(buffer, len);
  if (len > 0) {
    str.assign(buffer, len);
    buffer += len;
  } else {
    str.clear();
  }
}

INLINE void read_tensor(const char*& buffer, torch::Tensor& tensor) {
  // read dtype
  int8_t tensor_dtype;
  read_data(buffer, tensor_dtype);
  auto dtype = static_cast<torch::ScalarType>(tensor_dtype);
  // read ndim
  uint64_t ndim;
  read_data(buffer, ndim);
  // read shape
  std::vector<int64_t> shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    int64_t dim_size;
    read_data(buffer, dim_size);
    shape[i] = static_cast<int64_t>(dim_size);
  }
  // read data_bytes
  uint64_t data_bytes;
  read_data(buffer, data_bytes);

  tensor = torch::from_blob(const_cast<void*>(static_cast<const void*>(buffer)),
                            shape,
                            torch::TensorOptions()
                                .dtype(dtype)
                                .device(torch::kCPU)
                                .pinned_memory(true));
  buffer += data_bytes;
}

template <typename T>
INLINE void read_vector(const char*& buffer, std::vector<T>& vec) {
  uint64_t size;
  read_data(buffer, size);
  vec.resize(size);
  if (size > 0) {
    const size_t bytes = size * type_size<T>;
    std::memcpy(vec.data(), buffer, bytes);
    buffer += bytes;
  }
}

template <typename T>
INLINE void read_2d_vector(const char*& buffer,
                           std::vector<std::vector<T>>& vec2d) {
  uint64_t size;
  read_data(buffer, size);
  vec2d.resize(size);
  for (auto& vec : vec2d) {
    read_vector(buffer, vec);
  }
}

INLINE void read_sampling_param(const char*& buffer,
                                RequestSamplingParam& param) {
  const char* ptr = buffer;
  param.frequency_penalty = *reinterpret_cast<const float*>(ptr);
  ptr += type_size<float>;
  param.presence_penalty = *reinterpret_cast<const float*>(ptr);
  ptr += type_size<float>;
  param.repetition_penalty = *reinterpret_cast<const float*>(ptr);
  ptr += type_size<float>;
  param.temperature = *reinterpret_cast<const float*>(ptr);
  ptr += type_size<float>;
  param.top_p = *reinterpret_cast<const float*>(ptr);
  ptr += type_size<float>;
  param.top_k = *reinterpret_cast<const int64_t*>(ptr);
  ptr += type_size<int64_t>;
  param.logprobs = *reinterpret_cast<const bool*>(ptr);
  ptr += type_size<bool>;
  param.top_logprobs = *reinterpret_cast<const int64_t*>(ptr);
  ptr += type_size<int64_t>;
  param.do_sample = *reinterpret_cast<const bool*>(ptr);
  ptr += type_size<bool>;
  param.is_embeddings = *reinterpret_cast<const bool*>(ptr);
  ptr += type_size<bool>;
  param.beam_width = *reinterpret_cast<const int32_t*>(ptr);
  ptr += type_size<int32_t>;
  buffer = ptr;
}

INLINE void read_instance_info(const char*& buffer, InstanceInfo& info) {
  read_string(buffer, info.name);
  read_string(buffer, info.rpc_address);
  read_string(buffer, info.type);

  read_vector(buffer, info.cluster_ids);

  uint64_t addr_count;
  read_data(buffer, addr_count);
  info.addrs.resize(addr_count);
  for (auto& addr : info.addrs) {
    read_string(buffer, addr);
  }

  read_vector(buffer, info.k_cache_ids);
  read_vector(buffer, info.v_cache_ids);
  read_data(buffer, info.dp_size);

  uint64_t prof_size;
  read_data(buffer, prof_size);
  info.ttft_profiling_data.resize(prof_size);
  if (prof_size > 0) {
    std::memcpy(info.ttft_profiling_data.data(),
                buffer,
                prof_size * sizeof(std::pair<int32_t, int64_t>));
    buffer += prof_size * sizeof(std::pair<int32_t, int64_t>);
  }
}

INLINE void read_transfer_kv_info(const char*& buffer, TransferKVInfo& info) {
  read_string(buffer, info.request_id);
  read_vector(buffer, info.local_blocks_ids);
  read_vector(buffer, info.remote_blocks_ids);
  read_data(buffer, info.dp_rank);
  read_instance_info(buffer, info.remote_instance_info);
}

INLINE void read_eplb_info(const char*& buffer, EplbInfo& info) {
  read_data(buffer, info.prepare_layer_id);
  read_vector(buffer, info.expert_ids);
  read_data(buffer, info.update_layer_id);
}

INLINE void read_swap_blocks(const char*& buffer,
                             std::vector<BlockTransferInfo>& blocks) {
  uint64_t size;
  read_data(buffer, size);
  blocks.reserve(size);
  for (int i = 0; i < size; i++) {
    blocks.emplace_back(*reinterpret_cast<const int32_t*>(buffer),
                        *reinterpret_cast<const int32_t*>(buffer + 4));
  }
}

INLINE void read_mm_batch_data(const char*& buffer, MMBatchData& mm_data) {
  size_t size;
  read_data(buffer, size);
  uint32_t mm_type;
  read_data(buffer, mm_type);
  int32_t tensor_num;

  MMDict mm_dict;
  while (size--) {
    std::string mm_key;
    read_string(buffer, mm_key);
    read_data(buffer, tensor_num);
    if (tensor_num == 1) {
      torch::Tensor tensor;
      read_tensor(buffer, tensor);
      mm_dict[mm_key] = tensor;
    } else {
      std::vector<torch::Tensor> tensor_vec(tensor_num);
      for (size_t i = 0; i < tensor_num; ++i) {
        read_tensor(buffer, tensor_vec[i]);
      }
      mm_dict[mm_key] = tensor_vec;
    }
  }
  mm_data = std::move(MMBatchData(mm_type, mm_dict));
}

INLINE void deserialize_raw_forward_input(
    const char*& buffer,
    RawForwardInput& input,
    std::vector<RequestSamplingParam>& tmp_sampling_params) {
  read_vector(buffer, input.flatten_tokens_vec);
  read_vector(buffer, input.flatten_positions_vec);

  uint64_t sp_count;
  read_data(buffer, sp_count);
  input.sampling_params.reserve(sp_count);
  tmp_sampling_params.resize(sp_count);
  for (size_t i = 0; i < sp_count; ++i) {
    read_sampling_param(buffer, tmp_sampling_params[i]);
    input.sampling_params.push_back(&tmp_sampling_params[i]);
  }

  read_vector(buffer, input.selected_token_idxes);
  read_vector(buffer, input.sample_idxes);
  read_vector(buffer, input.unique_token_lens_vec);
  read_vector(buffer, input.seq_lens);
  read_vector(buffer, input.q_seq_lens);
  read_vector(buffer, input.new_token_slot_ids);
  read_vector(buffer, input.dp_global_token_nums);
  read_vector(buffer, input.embedding_ids);
  read_vector(buffer, input.src_block_indices);
  read_vector(buffer, input.dst_block_indices);
  read_vector(buffer, input.cum_sum);
  read_vector(buffer, input.new_cache_slot_offsets);
  read_vector(buffer, input.kv_cache_start_offsets);
  read_vector(buffer, input.extra_token_ids);
  read_vector(buffer, input.acc_logprob_vec);

  read_2d_vector(buffer, input.unique_token_ids_vec);
  read_2d_vector(buffer, input.unique_token_counts_vec);
  read_2d_vector(buffer, input.block_tables_vec);
  read_2d_vector(buffer, input.embeddings);

  uint64_t transfer_count;
  read_data(buffer, transfer_count);
  input.transfer_kv_infos.resize(transfer_count);
  for (auto& transfer : input.transfer_kv_infos) {
    read_transfer_kv_info(buffer, transfer);
  }

  read_swap_blocks(buffer, input.swap_blocks);
  read_data(buffer, input.batch_id);

  read_data(buffer, input.empty_kv_cache);
  read_data(buffer, input.global_empty_kv_cache);
  int32_t batch_forward_type;
  read_data(buffer, batch_forward_type);
  input.batch_forward_type = BatchForwardType(batch_forward_type);
  read_data(buffer, input.max_seq_len);
  read_data(buffer, input.q_max_seq_len);
  read_data(buffer, input.num_sequences);
  read_eplb_info(buffer, input.eplb_info);
  read_2d_vector(buffer, input.m_positions_vec);
  read_mm_batch_data(buffer, input.mm_data);
  read_vector(buffer, input.dp_is_decode);
}

INLINE void serialize_raw_forward_input(const RawForwardInput& input,
                                        char*& buffer) {
  write_vector(buffer, input.flatten_tokens_vec);
  write_vector(buffer, input.flatten_positions_vec);

  const uint64_t sp_count = input.sampling_params.size();
  write_data(buffer, sp_count);

  for (const auto* sp : input.sampling_params) {
    write_sampling_param(buffer, *sp);
  }

  write_vector(buffer, input.selected_token_idxes);
  write_vector(buffer, input.sample_idxes);
  write_vector(buffer, input.unique_token_lens_vec);
  write_vector(buffer, input.seq_lens);
  write_vector(buffer, input.q_seq_lens);
  write_vector(buffer, input.new_token_slot_ids);
  write_vector(buffer, input.dp_global_token_nums);
  write_vector(buffer, input.embedding_ids);
  write_vector(buffer, input.src_block_indices);
  write_vector(buffer, input.dst_block_indices);
  write_vector(buffer, input.cum_sum);
  write_vector(buffer, input.new_cache_slot_offsets);
  write_vector(buffer, input.kv_cache_start_offsets);
  write_vector(buffer, input.extra_token_ids);
  write_vector(buffer, input.acc_logprob_vec);

  write_2d_vector(buffer, input.unique_token_ids_vec);
  write_2d_vector(buffer, input.unique_token_counts_vec);
  write_2d_vector(buffer, input.block_tables_vec);
  write_2d_vector(buffer, input.embeddings);

  write_data(buffer, (uint64_t)input.transfer_kv_infos.size());
  for (const auto& t : input.transfer_kv_infos) {
    write_transfer_kv_info(buffer, t);
  }

  write_swap_blocks(buffer, input.swap_blocks);
  write_data(buffer, input.batch_id);

  write_data(buffer, input.empty_kv_cache);
  write_data(buffer, input.global_empty_kv_cache);
  write_data(buffer, input.batch_forward_type.value());
  write_data(buffer, input.max_seq_len);
  write_data(buffer, input.q_max_seq_len);
  write_data(buffer, input.num_sequences);
  write_eplb_info(buffer, input.eplb_info);
  write_2d_vector(buffer, input.m_positions_vec);
  write_mm_batch_data(buffer, input.mm_data);
  write_vector(buffer, input.dp_is_decode);
}

size_t calculate_raw_token_size(const RawToken& token) {
  size_t size = type_size<int64_t>;  // id

  size += type_size<bool>;
  if (token.logprob.has_value()) {
    size += type_size<float>;
  }

  size += type_size<uint64_t> + token.top_tokens.size() * type_size<int64_t>;
  size += type_size<uint64_t> + token.top_logprobs.size() * type_size<float>;
  size += type_size<uint64_t> + token.embeddings.size() * type_size<float>;

  return size;
}

size_t calculate_raw_sample_output_size(const RawSampleOutput& sample) {
  size_t size = type_size<uint64_t>;
  for (const auto& token : sample.tokens) {
    size += calculate_raw_token_size(token);
  }
  return size;
}

size_t calculate_raw_forward_output_size(const RawForwardOutput& output) {
  size_t size = 0;

  size += type_size<uint64_t>;
  for (const auto& sample : output.outputs) {
    size += calculate_raw_sample_output_size(sample);
  }

  size += get_vector_size(output.expert_load_data);
  size += get_vector_size(output.src_seq_idxes);
  size += get_vector_size(output.out_tokens);
  size += get_vector_size(output.out_logprobs);
  size += type_size<int32_t>;  // prepared_layer_id

  return size;
}

void write_raw_token(char*& buffer, const RawToken& token) {
  write_data(buffer, token.id);

  write_data(buffer, token.logprob.has_value());
  if (token.logprob.has_value()) {
    write_data(buffer, token.logprob.value());
  }

  write_vector(buffer, token.top_tokens);
  write_vector(buffer, token.top_logprobs);
  write_vector(buffer, token.embeddings);
}

void write_raw_sample_output(char*& buffer, const RawSampleOutput& sample) {
  write_data(buffer, static_cast<uint64_t>(sample.tokens.size()));
  for (const auto& token : sample.tokens) {
    write_raw_token(buffer, token);
  }
}

void read_raw_token(const char*& buffer, RawToken& token) {
  read_data(buffer, token.id);

  bool has_logprob;
  read_data(buffer, has_logprob);
  if (has_logprob) {
    float logprob_val;
    read_data(buffer, logprob_val);
    token.logprob = logprob_val;
  } else {
    token.logprob = std::nullopt;
  }

  read_vector(buffer, token.top_tokens);
  read_vector(buffer, token.top_logprobs);
  read_vector(buffer, token.embeddings);
}

void read_raw_sample_output(const char*& buffer, RawSampleOutput& sample) {
  uint64_t token_count;
  read_data(buffer, token_count);
  sample.tokens.resize(token_count);
  for (auto& token : sample.tokens) {
    read_raw_token(buffer, token);
  }
}

void deserialize_raw_forward_output(const char* buffer,
                                    RawForwardOutput& output) {
  uint64_t outputs_count;
  read_data(buffer, outputs_count);
  output.outputs.resize(outputs_count);
  for (auto& sample : output.outputs) {
    read_raw_sample_output(buffer, sample);
  }

  read_vector(buffer, output.expert_load_data);

  read_data(buffer, output.prepared_layer_id);
}

void serialize_raw_forward_output(const RawForwardOutput& output,
                                  char*& buffer) {
  write_data(buffer, static_cast<uint64_t>(output.outputs.size()));
  for (const auto& sample : output.outputs) {
    write_raw_sample_output(buffer, sample);
  }

  write_vector(buffer, output.expert_load_data);

  write_data(buffer, output.prepared_layer_id);
}

ForwardSharedMemoryManager::ForwardSharedMemoryManager(const std::string& name,
                                                       size_t size,
                                                       bool& is_creator,
                                                       ForwardType type)
    : SharedMemoryManager(name, size, is_creator), forward_type_(type) {
  control_ptr_ = static_cast<ControlMetadata*>(base_address());
  metadata_addr_ = static_cast<char*>(base_address()) + sizeof(ControlMetadata);
}

ForwardSharedMemoryManager::~ForwardSharedMemoryManager() = default;

/* The shared memory filename may have duplicates when using kill -9 xllm, but
  this doesn't affect usage.*/
std::string ForwardSharedMemoryManager::create_unique_name(
    const std::string& prefix,
    int dp_group,
    int forward_type,
    int rank) {
  std::string filename = prefix;
  if (forward_type == FORWARD_PB_INPUT_TYPE ||
      forward_type == FORWARD_RAW_INPUT_TYPE) {
    filename += "_dpg_" + std::to_string(dp_group) + "_input";
  } else if (forward_type == FORWARD_PB_OUTPUT_TYPE ||
             forward_type == FORWARD_RAW_OUTPUT_TYPE) {
    filename += "_rank_" + std::to_string(rank) + "_output";
  } else {
    // TODO: support more type later
  }

  return filename;
}

bool ForwardSharedMemoryManager::raw_input_write(
    const std::vector<RawForwardInput>& inputs) {
  uint64_t total_size = sizeof(ControlMetadata);
  for (const auto& input : inputs) {
    total_size += calculate_raw_forward_input_size(input);
  }
  if (unlikely(total_size > size())) {
    LOG(ERROR) << "raw input size overflow, total_size: " << total_size
               << ", shm size: " << size();
    return false;
  }

  char* data_ptr = static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  write_data(data_ptr, static_cast<uint64_t>(inputs.size()));
  for (const auto& input : inputs) {
    serialize_raw_forward_input(input, data_ptr);
  }
  std::atomic_thread_fence(std::memory_order_release);
  control_ptr_->version = ++last_version_;

  return true;
}

void convert_raw_forward_input_to_forward_input(RawForwardInput& raw_input,
                                                ForwardInput& forward_input) {
  auto tensor_options = torch::TensorOptions()
                            .dtype(torch::kInt)
                            .device(torch::kCPU)
                            .pinned_memory(true);

  forward_input.token_ids =
      torch::tensor(std::move(raw_input.flatten_tokens_vec), tensor_options);
  if (raw_input.flatten_positions_vec.size() > 0) {
    forward_input.positions = torch::tensor(
        std::move(raw_input.flatten_positions_vec), tensor_options);
  } else if (raw_input.m_positions_vec.size() > 0) {
    forward_input.positions =
        create_2d_tensor(std::move(raw_input.m_positions_vec), torch::kInt);
  }

  auto& input_params = forward_input.input_params;
  input_params.empty_kv_cache = raw_input.empty_kv_cache;
  input_params.global_empty_kv_cache = raw_input.global_empty_kv_cache;
  input_params.batch_forward_type = raw_input.batch_forward_type;
  input_params.num_sequences = raw_input.num_sequences;
  input_params.kv_max_seq_len = raw_input.max_seq_len;
  input_params.q_max_seq_len = raw_input.q_max_seq_len;
  input_params.embedding_ids = std::move(raw_input.embedding_ids);
  input_params.dp_global_token_nums = std::move(raw_input.dp_global_token_nums);
  input_params.dp_is_decode = std::move(raw_input.dp_is_decode);

  input_params.kv_seq_lens =
      torch::tensor(std::move(raw_input.seq_lens), tensor_options);
  input_params.q_seq_lens =
      torch::tensor(std::move(raw_input.q_seq_lens), tensor_options);
  input_params.kv_seq_lens_vec = std::move(raw_input.seq_lens);
  input_params.q_seq_lens_vec = std::move(raw_input.q_seq_lens);

  input_params.new_cache_slots =
      torch::tensor(std::move(raw_input.new_token_slot_ids), tensor_options);

  util::pad_2d_vector(raw_input.block_tables_vec, 0);
  input_params.block_tables =
      create_2d_tensor(std::move(raw_input.block_tables_vec), torch::kInt);

  input_params.src_block_indices =
      torch::tensor(std::move(raw_input.src_block_indices), tensor_options);
  input_params.dst_block_indices =
      torch::tensor(std::move(raw_input.dst_block_indices), tensor_options);
  input_params.cum_sum =
      torch::tensor(std::move(raw_input.cum_sum), tensor_options);

  input_params.swap_blocks = std::move(raw_input.swap_blocks);
  input_params.batch_id = std::move(raw_input.batch_id);
  input_params.extra_token_ids = std::move(raw_input.extra_token_ids);

  input_params.new_cache_slot_offsets = torch::tensor(
      std::move(raw_input.new_cache_slot_offsets), tensor_options);
  input_params.kv_cache_start_offsets = torch::tensor(
      std::move(raw_input.kv_cache_start_offsets), tensor_options);

  input_params.mm_data = std::move(raw_input.mm_data);
  if (!raw_input.selected_token_idxes.empty()) {
    util::pad_2d_vector<int64_t>(raw_input.unique_token_ids_vec, 0);
    util::pad_2d_vector(raw_input.unique_token_counts_vec, 0);
    forward_input.sampling_params.init(
        std::move(raw_input.sampling_params),
        std::move(raw_input.selected_token_idxes),
        std::move(raw_input.sample_idxes),
        std::move(raw_input.unique_token_ids_vec),
        std::move(raw_input.unique_token_counts_vec),
        std::move(raw_input.unique_token_lens_vec));
  }

  forward_input.acc_logprob = torch::tensor(
      std::move(raw_input.acc_logprob_vec),
      torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
  forward_input.transfer_kv_infos = std::move(raw_input.transfer_kv_infos);
  forward_input.eplb_info = std::move(raw_input.eplb_info);
}

void ForwardSharedMemoryManager::raw_input_read(
    std::vector<ForwardInput>& inputs) {
  while (true) {
    if (control_ptr_->version != last_version_) {
      last_version_ = control_ptr_->version;
      break;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(NUM_WAIT_NANOSECONDS));
  }

  const char* data_ptr =
      static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  uint64_t count;
  read_data(data_ptr, count);

  std::vector<std::vector<RequestSamplingParam>> tmp_sampling_params;
  std::vector<RawForwardInput> raw_inputs;
  tmp_sampling_params.resize(count);
  raw_inputs.resize(count);
  for (uint64_t i = 0; i < count; ++i) {
    deserialize_raw_forward_input(
        data_ptr, raw_inputs[i], tmp_sampling_params[i]);
  }

  // convert raw forward input to forward input
  inputs.resize(raw_inputs.size());
  for (uint64_t i = 0; i < count; ++i) {
    convert_raw_forward_input_to_forward_input(raw_inputs[i], inputs[i]);
  }

  return;
}

void convert_tensor_to_raw_output(const torch::Tensor& next_tokens,
                                  const torch::Tensor& logprobs,
                                  const torch::Tensor& top_tokens,
                                  const torch::Tensor& top_logprobs,
                                  const torch::Tensor& embeddings,
                                  const torch::Tensor& expert_load_data,
                                  int32_t prepared_layer_id,
                                  const torch::Tensor& src_seq_idxes,
                                  const torch::Tensor& out_tokens,
                                  const torch::Tensor& out_logprobs,
                                  RawForwardOutput& raw_output) {
  raw_output.prepared_layer_id = prepared_layer_id;

  if (FLAGS_enable_eplb) {
    torch::Tensor expert_load_data_flattened =
        expert_load_data.view({-1}).contiguous();
    if (expert_load_data_flattened.defined()) {
      const int64_t* data_ptr = expert_load_data_flattened.data_ptr<int64_t>();
      size_t size = static_cast<size_t>(expert_load_data_flattened.size(0));
      raw_output.expert_load_data.assign(data_ptr, data_ptr + size);
    }
  }

  if (src_seq_idxes.defined() && src_seq_idxes.numel() > 0) {
    const int32_t* data_ptr = src_seq_idxes.data_ptr<int32_t>();
    size_t size = static_cast<size_t>(src_seq_idxes.size(0));
    raw_output.src_seq_idxes.assign(data_ptr, data_ptr + size);
  }

  if (out_tokens.defined() && out_tokens.numel() > 0) {
    const int32_t* data_ptr = out_tokens.data_ptr<int32_t>();
    size_t size = static_cast<size_t>(out_tokens.size(0));
    raw_output.out_tokens.assign(data_ptr, data_ptr + size);
  }

  if (out_logprobs.defined() && out_logprobs.numel() > 0) {
    const float* data_ptr = out_logprobs.data_ptr<float>();
    size_t size = static_cast<size_t>(out_logprobs.size(0));
    raw_output.out_logprobs.assign(data_ptr, data_ptr + size);
  }

  int32_t num_seqs =
      next_tokens.defined() ? static_cast<int32_t>(next_tokens.size(0)) : 0;
  if (embeddings.defined() && embeddings.numel() > 0) {
    num_seqs = std::max(num_seqs, static_cast<int32_t>(embeddings.size(0)));
  }

  raw_output.outputs.reserve(num_seqs);
  for (int32_t output_idx = 0; output_idx < num_seqs; ++output_idx) {
    RawSampleOutput raw_sample_output;

    if (next_tokens.defined() && next_tokens.dim() == 2) {
      const auto curr_idx = output_idx;
      const auto curr_next_tokens = next_tokens[curr_idx];
      const auto curr_logprobs =
          logprobs.defined() ? logprobs[curr_idx] : logprobs;
      const auto curr_top_tokens =
          top_tokens.defined() ? top_tokens[curr_idx] : top_tokens;
      const auto curr_top_logprobs =
          top_logprobs.defined() ? top_logprobs[curr_idx] : top_logprobs;
      const auto curr_embeddings =
          embeddings.defined() ? embeddings[curr_idx] : embeddings;

      int32_t num_tokens = curr_next_tokens.size(0);
      raw_sample_output.tokens.reserve(num_tokens);

      for (int32_t i = 0; i < num_tokens; ++i) {
        const Token token = build_token(i,
                                        curr_next_tokens,
                                        curr_logprobs,
                                        curr_top_tokens,
                                        curr_top_logprobs);
        if (token.id == -1) {
          break;
        }

        RawToken raw_token;
        raw_token.id = token.id;
        raw_token.logprob = token.logprob;
        raw_token.top_tokens = token.top_tokens;
        raw_token.top_logprobs = token.top_logprobs;

        if (curr_embeddings.defined()) {
          const auto token_embeddings = curr_embeddings[i];
          if (token_embeddings.defined()) {
            const float* emb_ptr = token_embeddings.data_ptr<float>();
            size_t emb_size = static_cast<size_t>(token_embeddings.size(0));
            raw_token.embeddings.assign(emb_ptr, emb_ptr + emb_size);
          }
        }

        raw_sample_output.tokens.push_back(std::move(raw_token));
      }
    } else {
      RawToken raw_token;

      if (next_tokens.defined() && next_tokens.numel() > 0) {
        const Token token = build_token(
            output_idx, next_tokens, logprobs, top_tokens, top_logprobs);
        raw_token.id = token.id;
        raw_token.logprob = token.logprob;
        raw_token.top_tokens = std::move(token.top_tokens);
        raw_token.top_logprobs = std::move(token.top_logprobs);
      } else {
        raw_token.id = -1;
        raw_token.logprob = std::nullopt;
      }

      if (embeddings.defined()) {
        const auto token_embeddings = embeddings[output_idx];
        if (token_embeddings.defined()) {
          const float* emb_ptr = token_embeddings.data_ptr<float>();
          size_t emb_size = static_cast<size_t>(token_embeddings.size(0));
          raw_token.embeddings.assign(emb_ptr, emb_ptr + emb_size);
        }
      }

      raw_sample_output.tokens.push_back(std::move(raw_token));
    }
    raw_output.outputs.push_back(std::move(raw_sample_output));
  }
}

bool ForwardSharedMemoryManager::raw_output_write(
    const torch::Tensor& next_tokens,
    const torch::Tensor& logprobs,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    const torch::Tensor& embeddings,
    const torch::Tensor& expert_load_data,
    int32_t prepared_layer_id,
    const torch::Tensor& src_seq_idxes,
    const torch::Tensor& out_tokens,
    const torch::Tensor& out_logprobs) {
  RawForwardOutput output;
  convert_tensor_to_raw_output(next_tokens,
                               logprobs,
                               top_tokens,
                               top_logprobs,
                               embeddings,
                               expert_load_data,
                               prepared_layer_id,
                               src_seq_idxes,
                               out_tokens,
                               out_logprobs,
                               output);
  uint64_t total_size = sizeof(ControlMetadata);
  total_size += calculate_raw_forward_output_size(output);
  if (unlikely(total_size > size())) {
    LOG(ERROR) << "raw output size overflow, total_size: " << total_size
               << ", shm size: " << size();
    return false;
  }

  char* data_ptr = static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  serialize_raw_forward_output(output, data_ptr);
  char* test = static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  std::atomic_thread_fence(std::memory_order_release);
  control_ptr_->version = ++last_version_;

  return true;
}

void ForwardSharedMemoryManager::raw_output_read(RawForwardOutput& output) {
  while (true) {
    if (control_ptr_->version != last_version_) {
      last_version_ = control_ptr_->version;
      break;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(NUM_WAIT_NANOSECONDS));
  }

  const char* data_ptr =
      static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  char* test = static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  deserialize_raw_forward_output(data_ptr, output);

  return;
}

void ForwardSharedMemoryManager::clear() {
  std::memset(base_address(), 0, size());
}
}  // namespace xllm
