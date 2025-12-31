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
#include "core/util/tensor_helper.h"
#include "util/utils.h"

#if defined(__GNUC__)
static inline bool(likely)(bool x) { return __builtin_expect((x), true); }
static inline bool(unlikely)(bool x) { return __builtin_expect((x), false); }
#else
static inline bool(likely)(bool x) { return x; }
static inline bool(unlikely)(bool x) { return x; }
#endif

namespace xllm {

namespace {
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

inline size_t get_string_size(const std::string& str) {
  return type_size<uint64_t> + str.size();
}

template <typename T>
inline size_t get_vector_size(const std::vector<T>& vec) {
  return type_size<uint64_t> + vec.size() * type_size<T>;
}

template <typename T>
size_t get_vector_to_tensor_size(const std::vector<T>& vec) {
  uint64_t size = type_size<uint64_t>;  // ndim
  if (vec.size() == 0) {
    return size;
  }
  size += type_size<uint64_t>;        // shape
  size += type_size<int8_t>;          // dtype
  size += type_size<uint64_t>;        // databytes
  size += vec.size() * type_size<T>;  // data
  return size;
}

inline size_t get_tensor_size(const torch::Tensor& tensor) {
  uint64_t size = type_size<uint64_t>;  // ndim
  if (!tensor.defined()) {
    return size;
  }
  size += type_size<uint64_t> * tensor.dim();      // shape
  size += type_size<int8_t>;                       // dtype
  size += type_size<uint64_t>;                     // databytes
  size += tensor.numel() * tensor.element_size();  // data
  return size;
}

template <typename T>
inline size_t get_2d_vector_size(const std::vector<std::vector<T>>& vec2d) {
  size_t size = type_size<uint64_t>;
  for (const auto& vec : vec2d) {
    size += get_vector_size(vec);
  }
  return size;
}

template <typename T>
size_t get_2d_vector_to_tensor_size(const std::vector<std::vector<T>>& vec2d) {
  uint64_t size = type_size<uint64_t>;  // ndim
  if (vec2d.size() == 0 || vec2d[0].size() == 0) {
    return size;
  }
  size += type_size<uint64_t> * 2;                        // shape
  size += type_size<int8_t>;                              // dtype
  size += type_size<uint64_t>;                            // databytes
  size += vec2d.size() * vec2d[0].size() * type_size<T>;  // data
  return size;
}

inline size_t get_instance_info_size(const InstanceInfo& info) {
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

inline size_t get_transfer_kv_info_size(const TransferKVInfo& info) {
  return get_string_size(info.request_id) +
         get_vector_size(info.local_blocks_ids) +
         get_vector_size(info.remote_blocks_ids) +
         type_size<int32_t>  // dp_rank
         + get_instance_info_size(info.remote_instance_info);
}

inline size_t get_eplb_info_size(const EplbInfo& info) {
  return type_size<int32_t>  // prepare_layer_id
         + get_vector_size(info.expert_ids) +
         type_size<int32_t>;  // update_layer_id
}

inline size_t get_mm_dict_size(const MMDict& mm_dict) {
  size_t total = 0;
  total += type_size<size_t>;  // mm_dict size
  for (auto& [mm_key, mm_value] : mm_dict) {
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

inline size_t get_mm_item_size(const MMDataItem& mm_item) {
  size_t total = 0;

  total += type_size<uint32_t>;               // type
  total += get_mm_dict_size(mm_item.data());  // dict

  // token_pos
  total += type_size<uint32_t> * 2;

  // prefix_cache
  total += MURMUR_HASH3_VALUE_LEN;
  total += type_size<uint32_t>;

  return total;
}

size_t get_sampling_params_size(const SamplingParameters& params) {
  size_t total = 0;

  total += get_tensor_size(params.selected_token_idxes);
  total += get_tensor_size(params.frequency_penalties);
  total += get_tensor_size(params.presence_penalties);
  total += get_tensor_size(params.repetition_penalties);
  total += get_tensor_size(params.temperatures);
  total += get_tensor_size(params.top_p);
  total += get_tensor_size(params.top_k);
  total += get_tensor_size(params.unique_token_ids);
  total += get_tensor_size(params.unique_token_counts);
  total += get_tensor_size(params.unique_token_ids_lens);
  total += get_tensor_size(params.sample_idxes);
  total += get_tensor_size(params.do_sample);
  total += type_size<bool> * 5    // all_random_sample + all_greedy_sample +
                                  // logprobs + is_embeddings + use_beam_search
           + type_size<int64_t>;  // max_top_logprobs
  return total;
}

inline size_t get_mm_data_size(const MMData& mm_data) {
  size_t total = 0;
  total += type_size<uint32_t>;  //  mm_type
  if (mm_data.hold<MMItemVec>()) {
    total += type_size<size_t>;  // num of mm_items
    const auto& mm_items = mm_data.items<MMItemVec>();
    for (const auto& mm_item : mm_items) {
      total += get_mm_item_size(mm_item);
    }
  } else if (mm_data.hold<MMDict>()) {
    total += get_mm_dict_size(mm_data.items<MMDict>());
  }
  return total;
}

inline size_t get_mm_batch_data_size(const MMBatchData& mm_data) {
  const auto& vec = mm_data.mm_data_vec();

  size_t total = 0;
  total += type_size<size_t>;   // num of vec
  total += type_size<uint8_t>;  // is_mm_item
  for (const auto& mm_data : vec) {
    total += get_mm_data_size(mm_data);
  }
  return total;
}

size_t calculate_raw_forward_input_size(const RawForwardInput& input) {
  size_t total = 0;

  // flatten_tokens_vec
  total += get_vector_to_tensor_size(input.flatten_tokens_vec);
  if (input.flatten_positions_vec.size() > 0) {
    // flatten_positions_vec
    total += get_vector_to_tensor_size(input.flatten_positions_vec);
  } else {
    // m_positions_vec
    total += get_2d_vector_to_tensor_size(input.m_positions_vec);
  }

  // ModelInputParams
  total += type_size<bool> * 2        // empty_kv_cache + global_empty_kv_cache
           + type_size<int32_t>       // batch_forward_type
           + type_size<int32_t>       // num_sequences
           + type_size<uint32_t> * 2  // kv_max_seq_len + q_max_seq_len
           + type_size<uint64_t>;     // batch_id
  total += get_vector_to_tensor_size(input.q_seq_lens);
  total += get_vector_to_tensor_size(input.seq_lens);
  total += get_vector_to_tensor_size(input.new_token_slot_ids);
  total += get_2d_vector_to_tensor_size(input.block_tables_vec);
  total += get_vector_to_tensor_size(input.paged_kv_indptr);
  total += get_vector_to_tensor_size(input.paged_kv_indices);
  total += get_vector_to_tensor_size(input.paged_kv_last_page_len);
  total += get_vector_to_tensor_size(input.new_cache_slot_offsets);
  total += get_vector_to_tensor_size(input.kv_cache_start_offsets);
  total += get_2d_vector_to_tensor_size(input.embeddings);
  total += get_vector_size(input.dp_global_token_nums);
  total += get_vector_size(input.dp_is_decode);
  total += get_vector_size(input.embedding_ids);
  total += get_vector_size(input.extra_token_ids);
  total += type_size<uint64_t> +
           input.swap_blocks.size() * swap_block_info_fixed_size();
  total += get_vector_to_tensor_size(input.src_block_indices);
  total += get_vector_to_tensor_size(input.dst_block_indices);
  total += get_vector_to_tensor_size(input.cum_sum);
  total += get_mm_batch_data_size(input.mm_data);
  total += get_vector_to_tensor_size(input.kv_cache_tokens_nums);

  // SamplingParameters
  total += type_size<uint64_t>;  // selected_token_idxes.size()
  if (input.selected_token_idxes.size() > 0) {
    SamplingParameters sampling_params;
    sampling_params.init(input.sampling_params,
                         input.selected_token_idxes,
                         input.sample_idxes,
                         input.unique_token_ids_vec,
                         input.unique_token_counts_vec,
                         input.unique_token_lens_vec);
    total += get_sampling_params_size(sampling_params);
  }
  // acc_logprob
  total += get_vector_to_tensor_size(input.acc_logprob_vec);

  // transfer_kv_infos
  total += type_size<uint64_t>;
  for (const auto& t : input.transfer_kv_infos) {
    total += get_transfer_kv_info_size(t);
  }
  // eplb_info
  total += get_eplb_info_size(input.eplb_info);

  return total;
}

template <typename T>
inline void write_data(char*& buffer, const T& data) {
  *reinterpret_cast<T*>(buffer) = data;
  buffer += type_size<T>;
}

inline void write_string(char*& buffer, const std::string& str) {
  const uint64_t len = str.size();
  write_data(buffer, len);
  if (len > 0) {
    std::memcpy(buffer, str.data(), len);
    buffer += len;
  }
}

inline void write_tensor(char*& buffer, const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    uint64_t ndim = 0;
    write_data(buffer, ndim);
    return;
  }
  auto contig_tensor = tensor.cpu().contiguous();
  // write ndim
  const uint64_t tensor_ndim = contig_tensor.dim();
  write_data(buffer, tensor_ndim);
  // write shape
  for (int64_t i = 0; i < contig_tensor.dim(); ++i) {
    write_data(buffer, static_cast<uint64_t>(contig_tensor.size(i)));
  }
  // write dtype
  const int8_t tensor_dtype = static_cast<int8_t>(contig_tensor.scalar_type());
  write_data(buffer, tensor_dtype);
  // write data_bytes
  const uint64_t tensor_data_bytes =
      contig_tensor.numel() * contig_tensor.element_size();
  write_data(buffer, tensor_data_bytes);

  if (tensor_data_bytes > 0) {
    std::memcpy(buffer, contig_tensor.data_ptr(), tensor_data_bytes);
    buffer += tensor_data_bytes;
  }
}

inline void write_sampling_param(char*& buffer,
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
inline void write_vector(char*& buffer, const std::vector<T>& vec) {
  const uint64_t size = vec.size();
  write_data(buffer, size);
  if (size > 0) {
    const size_t bytes = size * type_size<T>;
    std::memcpy(buffer, vec.data(), bytes);
    buffer += bytes;
  }
}

template <typename T>
void write_vector_to_tensor(char*& buffer, const std::vector<T>& vec) {
  // write ndim
  uint64_t ndim;
  if (vec.empty()) {
    ndim = 0;
    write_data(buffer, ndim);
    return;
  }
  ndim = 1;
  write_data(buffer, ndim);
  // write shape
  write_data(buffer, vec.size());
  // write dtype
  const int8_t tensor_dtype = static_cast<int8_t>(get_scalar_type<T>());
  write_data(buffer, tensor_dtype);
  // write data_bytes
  const uint64_t data_bytes = vec.size() * type_size<T>;
  write_data(buffer, data_bytes);
  // write vec data
  std::memcpy(buffer, vec.data(), data_bytes);
  buffer += data_bytes;
}

template <typename T>
inline void write_2d_vector(char*& buffer,
                            const std::vector<std::vector<T>>& vec2d) {
  write_data(buffer, (uint64_t)vec2d.size());
  for (const auto& vec : vec2d) {
    write_vector(buffer, vec);
  }
}

template <typename T>
void write_2d_vector_to_tensor(char*& buffer,
                               const std::vector<std::vector<T>>& vec2d) {
  // write ndim
  uint64_t ndim;
  if (vec2d.size() == 0 || vec2d[0].size() == 0) {
    ndim = 0;
    write_data(buffer, ndim);
    return;
  }
  ndim = 2;
  write_data(buffer, ndim);
  // write shape
  write_data(buffer, vec2d.size());
  write_data(buffer, vec2d[0].size());
  // write dtype
  const int8_t tensor_dtype = static_cast<int8_t>(get_scalar_type<T>());
  write_data(buffer, tensor_dtype);
  // write data_bytes
  const uint64_t per_data_bytes = vec2d[0].size() * type_size<T>;
  const uint64_t data_bytes = vec2d.size() * per_data_bytes;
  write_data(buffer, data_bytes);
  // write vec data
  for (const auto& vec : vec2d) {
    std::memcpy(buffer, vec.data(), per_data_bytes);
    buffer += per_data_bytes;
  }
}

inline void write_instance_info(char*& buffer, const InstanceInfo& info) {
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

inline void write_transfer_kv_info(char*& buffer, const TransferKVInfo& info) {
  write_string(buffer, info.request_id);
  write_vector(buffer, info.local_blocks_ids);
  write_vector(buffer, info.remote_blocks_ids);
  write_data(buffer, info.dp_rank);
  write_instance_info(buffer, info.remote_instance_info);
}

inline void write_eplb_info(char*& buffer, const EplbInfo& info) {
  write_data(buffer, info.prepare_layer_id);
  write_vector(buffer, info.expert_ids);
  write_data(buffer, info.update_layer_id);
}

inline void write_swap_blocks(char*& buffer,
                              const std::vector<BlockTransferInfo>& blocks) {
  write_data(buffer, (uint64_t)blocks.size());

  for (const auto& b : blocks) {
    write_data(buffer, b.src_block_id);
    write_data(buffer, b.dst_block_id);
  }
}

inline void write_vector_tensor(char*& buffer,
                                const std::vector<torch::Tensor>& tensor_vec) {
  int32_t tensor_num = tensor_vec.size();
  write_data(buffer, tensor_num);
  for (const auto& tensor : tensor_vec) {
    write_tensor(buffer, tensor);
  }
}

inline void write_mm_dict(char*& buffer, const MMDict& mm_dict) {
  // size
  size_t size = mm_dict.size();
  write_data(buffer, (size_t)size);
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

inline void write_mm_item(char*& buffer, const MMDataItem& item) {
  write_data(buffer, item.type());
  write_mm_dict(buffer, item.data());

  const auto& state = item.state();
  // write token_pos
  write_data(buffer, state.token_pos().offset);
  write_data(buffer, state.token_pos().length);

  // write prefix_cache
  memcpy(buffer, state.prefix_cache().key.data, MURMUR_HASH3_VALUE_LEN);
  buffer += MURMUR_HASH3_VALUE_LEN;
  write_data(buffer, state.prefix_cache().cached_token_num);
}

inline void write_mm_data_items(char*& buffer, const MMData& mm_data) {
  const auto& mm_items = mm_data.items<MMItemVec>();
  write_data(buffer, mm_data.type());
  write_data(buffer, mm_items.size());
  for (const auto& mm_item : mm_items) {
    write_mm_item(buffer, mm_item);
  }
}

inline void write_mm_data_dict(char*& buffer, const MMData& mm_data) {
  const auto& mm_dict = mm_data.items<MMDict>();
  write_data(buffer, mm_data.type());
  write_mm_dict(buffer, mm_dict);
}

inline void write_mm_batch_data(char*& buffer, const MMBatchData& mm_data) {
  const auto& vec = mm_data.mm_data_vec();
  write_data(buffer, vec.size());

  uint8_t is_mm_item =
      vec.size() ? static_cast<uint8_t>(vec[0].hold<MMItemVec>()) : 1;
  write_data(buffer, is_mm_item);
  std::function<void(char*&, const MMData&)> write_mm_data =
      is_mm_item ? write_mm_data_items : write_mm_data_dict;
  for (const auto& mm_data : vec) {
    write_mm_data(buffer, mm_data);
  }
}

inline void safe_advance_buffer(const char*& buffer, size_t offset) {
  if (buffer != nullptr) {
    buffer += offset;
  }
}

template <typename T>
inline void read_data(const char*& buffer, T& data) {
  data = *reinterpret_cast<const T*>(buffer);
  buffer += type_size<T>;
}

template <typename T>
inline void read_data(const char*& buffer,
                      T& data,
                      const char*& device_buffer) {
  data = *reinterpret_cast<const T*>(buffer);
  buffer += type_size<T>;
  safe_advance_buffer(device_buffer, type_size<T>);
}

inline void read_string(const char*& buffer, std::string& str) {
  uint64_t len;
  read_data(buffer, len);
  if (len > 0) {
    str.assign(buffer, len);
    buffer += len;
  } else {
    str.clear();
  }
}

inline void read_string(const char*& buffer,
                        std::string& str,
                        const char*& device_buffer) {
  uint64_t len;
  read_data(buffer, len, device_buffer);
  if (len > 0) {
    str.assign(buffer, len);
    buffer += len;
    safe_advance_buffer(device_buffer, len);
  } else {
    str.clear();
  }
}

inline void read_tensor(const char*& buffer, torch::Tensor& tensor) {
  // read ndim
  uint64_t ndim;
  read_data(buffer, ndim);
  if (ndim == 0) {
    return;
  }
  // read shape
  std::vector<int64_t> shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    int64_t dim_size;
    read_data(buffer, dim_size);
    shape[i] = static_cast<int64_t>(dim_size);
  }
  // read dtype
  int8_t tensor_dtype;
  read_data(buffer, tensor_dtype);
  auto dtype = static_cast<torch::ScalarType>(tensor_dtype);
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

void read_tensor(const char*& buffer,
                 torch::Tensor& tensor,
                 const char*& device_buffer) {
  // read ndim
  uint64_t ndim;
  read_data(buffer, ndim, device_buffer);
  if (ndim == 0) {
    return;
  }
  // read shape
  std::vector<int64_t> shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    int64_t dim_size;
    read_data(buffer, dim_size, device_buffer);
    shape[i] = static_cast<int64_t>(dim_size);
  }
  // read dtype
  int8_t tensor_dtype;
  read_data(buffer, tensor_dtype, device_buffer);
  auto dtype = static_cast<torch::ScalarType>(tensor_dtype);
  // read data_bytes
  uint64_t data_bytes;
  read_data(buffer, data_bytes, device_buffer);

  if (device_buffer != nullptr) {
    tensor = get_tensor_from_blob(shape, dtype, device_buffer);
  } else {
    tensor =
        torch::from_blob(const_cast<void*>(static_cast<const void*>(buffer)),
                         shape,
                         torch::TensorOptions()
                             .dtype(dtype)
                             .device(torch::kCPU)
                             .pinned_memory(true));
  }
  buffer += data_bytes;
  safe_advance_buffer(device_buffer, data_bytes);
}

template <typename T>
inline void read_vector(const char*& buffer, std::vector<T>& vec) {
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
inline void read_vector(const char*& buffer,
                        std::vector<T>& vec,
                        const char*& device_buffer) {
  uint64_t size;
  read_data(buffer, size, device_buffer);
  vec.resize(size);
  if (size > 0) {
    const size_t bytes = size * type_size<T>;
    std::memcpy(vec.data(), buffer, bytes);
    buffer += bytes;
    safe_advance_buffer(device_buffer, bytes);
  }
}

template <typename T>
inline void read_tensor_and_vector(const char*& buffer,
                                   torch::Tensor& tensor,
                                   std::vector<T>& vec,
                                   const char*& device_buffer) {
  // read ndim
  uint64_t ndim;
  read_data(buffer, ndim, device_buffer);
  if (ndim == 0) {
    return;
  }
  // read shape
  std::vector<int64_t> shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    int64_t dim_size;
    read_data(buffer, dim_size, device_buffer);
    shape[i] = static_cast<int64_t>(dim_size);
  }
  vec.resize(shape[0]);
  // read dtype
  int8_t tensor_dtype;
  read_data(buffer, tensor_dtype, device_buffer);
  auto dtype = static_cast<torch::ScalarType>(tensor_dtype);
  // read data_bytes
  uint64_t data_bytes;
  read_data(buffer, data_bytes, device_buffer);

  if (device_buffer != nullptr) {
    tensor = get_tensor_from_blob(shape, dtype, device_buffer);
  } else {
    tensor =
        torch::from_blob(const_cast<void*>(static_cast<const void*>(buffer)),
                         shape,
                         torch::TensorOptions()
                             .dtype(dtype)
                             .device(torch::kCPU)
                             .pinned_memory(true));
  }
  std::memcpy(vec.data(), buffer, data_bytes);
  buffer += data_bytes;
  safe_advance_buffer(device_buffer, data_bytes);
}

template <typename T>
inline void read_2d_vector(const char*& buffer,
                           std::vector<std::vector<T>>& vec2d) {
  uint64_t size;
  read_data(buffer, size);
  vec2d.resize(size);
  for (auto& vec : vec2d) {
    read_vector(buffer, vec);
  }
}

inline void read_sampling_param(const char*& buffer,
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

inline void read_instance_info(const char*& buffer, InstanceInfo& info) {
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

inline void read_transfer_kv_info(const char*& buffer, TransferKVInfo& info) {
  read_string(buffer, info.request_id);
  read_vector(buffer, info.local_blocks_ids);
  read_vector(buffer, info.remote_blocks_ids);
  read_data(buffer, info.dp_rank);
  read_instance_info(buffer, info.remote_instance_info);
}

inline void read_eplb_info(const char*& buffer, EplbInfo& info) {
  read_data(buffer, info.prepare_layer_id);
  read_vector(buffer, info.expert_ids);
  read_data(buffer, info.update_layer_id);
}

inline void read_swap_blocks(const char*& buffer,
                             std::vector<BlockTransferInfo>& blocks,
                             const char*& device_buffer) {
  uint64_t size;
  read_data(buffer, size, device_buffer);
  blocks.reserve(size);

  int32_t src_block_id;
  int32_t dst_block_id;
  for (int i = 0; i < size; i++) {
    read_data(buffer, src_block_id, device_buffer);
    read_data(buffer, dst_block_id, device_buffer);
    blocks.emplace_back(src_block_id, dst_block_id);
  }
}

inline void read_vector_tensor(const char*& buffer,
                               std::vector<torch::Tensor>& tensor_vec) {
  int32_t tensor_num;
  read_data(buffer, tensor_num);
  tensor_vec.resize(tensor_num);
  for (size_t i = 0; i < tensor_num; ++i) {
    read_tensor(buffer, tensor_vec[i]);
  }
}

inline void read_mm_dict(const char*& buffer,
                         MMDict& mm_dict,
                         const char*& device_buffer) {
  size_t size;
  read_data(buffer, size, device_buffer);
  int32_t tensor_num;
  while (size--) {
    std::string mm_key;
    read_string(buffer, mm_key, device_buffer);
    read_data(buffer, tensor_num, device_buffer);
    if (tensor_num == 1) {
      torch::Tensor tensor;
      read_tensor(buffer, tensor, device_buffer);
      mm_dict[mm_key] = tensor;
    } else {
      std::vector<torch::Tensor> tensor_vec(tensor_num);
      for (size_t i = 0; i < tensor_num; ++i) {
        read_tensor(buffer, tensor_vec[i], device_buffer);
      }
      mm_dict[mm_key] = tensor_vec;
    }
  }
}

inline void read_mm_item(const char*& buffer,
                         MMDataItem& item,
                         const char*& device_buffer) {
  uint32_t type;
  read_data(buffer, type, device_buffer);
  MMDict dict;
  read_mm_dict(buffer, dict, device_buffer);
  auto mm_type_value = static_cast<MMType::Value>(type);
  item = std::move(MMDataItem(mm_type_value, dict));
  auto& state = item.mutable_state();

  // read token_pos
  read_data(buffer, state.mutable_token_pos().offset, device_buffer);
  read_data(buffer, state.mutable_token_pos().length, device_buffer);

  // read prefix_cache
  std::memcpy(
      state.mutable_prefix_cache().key.data, buffer, MURMUR_HASH3_VALUE_LEN);
  buffer += MURMUR_HASH3_VALUE_LEN;
  safe_advance_buffer(device_buffer, MURMUR_HASH3_VALUE_LEN);
  read_data(
      buffer, state.mutable_prefix_cache().cached_token_num, device_buffer);
}

inline void read_mm_data_dict(const char*& buffer,
                              MMData& mm_data,
                              const char*& device_buffer) {
  uint32_t mm_type;
  read_data(buffer, mm_type, device_buffer);
  MMDict mm_dict;
  read_mm_dict(buffer, mm_dict, device_buffer);
  MMType ty{static_cast<MMType::Value>(mm_type)};
  mm_data = MMData(ty, mm_dict);
}

inline void read_mm_data_items(const char*& buffer,
                               MMData& mm_data,
                               const char*& device_buffer) {
  uint32_t mm_type;
  read_data(buffer, mm_type, device_buffer);
  size_t mm_items_num;
  read_data(buffer, mm_items_num, device_buffer);
  MMItemVec mm_items;
  mm_items.reserve(mm_items_num);
  MMDataItem mm_item(MMType::NONE);
  for (size_t idx = 0; idx < mm_items_num; ++idx) {
    read_mm_item(buffer, mm_item, device_buffer);
    mm_items.push_back(std::move(mm_item));
  }
  MMType ty{static_cast<MMType::Value>(mm_type)};
  mm_data = MMData(ty, std::move(mm_items));
}

inline void read_mm_batch_data(const char*& buffer,
                               MMBatchData& batch_mm_data,
                               const char*& device_buffer) {
  std::vector<MMData> vec;

  size_t mm_data_num;
  read_data(buffer, mm_data_num, device_buffer);
  uint8_t is_mm_item;
  read_data(buffer, is_mm_item, device_buffer);
  vec.reserve(mm_data_num);
  MMData mm_data;
  std::function<void(const char*&, MMData&, const char*&)> read_mm_data =
      is_mm_item ? read_mm_data_items : read_mm_data_dict;
  for (size_t i = 0; i < mm_data_num; ++i) {
    read_mm_data(buffer, mm_data, device_buffer);
    vec.push_back(std::move(mm_data));
  }

  batch_mm_data.batch(std::move(vec));
}

inline void deserialize_raw_forward_input(const char*& buffer,
                                          const uint64_t buffer_size,
                                          ForwardInput& forward_input,
                                          const torch::Device& device) {
  const char* device_buffer = nullptr;
#if defined(USE_NPU)
  if (FLAGS_use_contiguous_input_buffer) {
    // h to d
    auto host_input_buffer =
        torch::from_blob(const_cast<char*>(buffer),
                         {static_cast<int64_t>(buffer_size)},
                         torch::dtype(torch::kUInt8));
    forward_input.device_input_buffer = host_input_buffer.to(device);
    device_buffer = (char*)forward_input.device_input_buffer.data_ptr();
  }
#endif

  read_tensor(buffer, forward_input.token_ids, device_buffer);
  read_tensor(buffer, forward_input.positions, device_buffer);

  // input_params
  auto& input_params = forward_input.input_params;
  read_data(buffer, input_params.empty_kv_cache, device_buffer);
  read_data(buffer, input_params.global_empty_kv_cache, device_buffer);
  int32_t batch_forward_type;
  read_data(buffer, batch_forward_type, device_buffer);
  input_params.batch_forward_type = BatchForwardType(batch_forward_type);
  read_data(buffer, input_params.num_sequences, device_buffer);
  read_data(buffer, input_params.kv_max_seq_len, device_buffer);
  read_data(buffer, input_params.q_max_seq_len, device_buffer);
  read_data(buffer, input_params.batch_id, device_buffer);
  read_tensor_and_vector(buffer,
                         input_params.q_seq_lens,
                         input_params.q_seq_lens_vec,
                         device_buffer);
  read_tensor_and_vector(buffer,
                         input_params.kv_seq_lens,
                         input_params.kv_seq_lens_vec,
                         device_buffer);
  read_tensor(buffer, input_params.paged_kv_indptr, device_buffer);
  read_tensor(buffer, input_params.paged_kv_indices, device_buffer);
  read_tensor(buffer, input_params.paged_kv_last_page_len, device_buffer);
  read_tensor(buffer, input_params.new_cache_slot_offsets, device_buffer);
  read_tensor(buffer, input_params.kv_cache_start_offsets, device_buffer);
  read_tensor(buffer, input_params.input_embedding, device_buffer);
  read_vector(buffer, input_params.dp_global_token_nums, device_buffer);
  read_vector(buffer, input_params.dp_is_decode, device_buffer);
  read_vector(buffer, input_params.embedding_ids, device_buffer);
  read_vector(buffer, input_params.extra_token_ids, device_buffer);
  read_swap_blocks(buffer, input_params.swap_blocks, device_buffer);
  read_tensor(buffer, input_params.src_block_indices, device_buffer);
  read_tensor(buffer, input_params.dst_block_indices, device_buffer);
  read_tensor(buffer, input_params.cum_sum, device_buffer);
  read_mm_batch_data(buffer, input_params.mm_data, device_buffer);
  read_tensor_and_vector(buffer,
                         input_params.kv_cache_tokens_nums,
                         input_params.kv_cache_tokens_nums_host,
                         device_buffer);

  // sampling_params
  uint64_t selected_token_idxes_size;
  read_data(buffer, selected_token_idxes_size, device_buffer);
  if (selected_token_idxes_size > 0) {
    auto& sampling_params = forward_input.sampling_params;
    read_tensor(buffer, sampling_params.selected_token_idxes, device_buffer);
    read_tensor(buffer, sampling_params.frequency_penalties, device_buffer);
    read_tensor(buffer, sampling_params.presence_penalties, device_buffer);
    read_tensor(buffer, sampling_params.repetition_penalties, device_buffer);
    read_tensor(buffer, sampling_params.temperatures, device_buffer);
    read_tensor(buffer, sampling_params.top_p, device_buffer);
    read_tensor(buffer, sampling_params.top_k, device_buffer);
    read_tensor(buffer, sampling_params.unique_token_ids, device_buffer);
    read_tensor(buffer, sampling_params.unique_token_counts, device_buffer);
    read_tensor(buffer, sampling_params.unique_token_ids_lens, device_buffer);
    read_tensor(buffer, sampling_params.sample_idxes, device_buffer);
    read_tensor(buffer, sampling_params.do_sample, device_buffer);
    read_data(buffer, sampling_params.all_random_sample, device_buffer);
    read_data(buffer, sampling_params.all_greedy_sample, device_buffer);
    read_data(buffer, sampling_params.logprobs, device_buffer);
    read_data(buffer, sampling_params.is_embeddings, device_buffer);
    read_data(buffer, sampling_params.max_top_logprobs, device_buffer);
    read_data(buffer, sampling_params.use_beam_search, device_buffer);
  }
  // acc_logprob
  read_tensor(buffer, forward_input.acc_logprob, device_buffer);

  // All inputs below are host data, no need to handle device-side pointers
  // transfer_kv_infos
  uint64_t transfer_count;
  read_data(buffer, transfer_count);
  forward_input.transfer_kv_infos.resize(transfer_count);
  for (auto& transfer : forward_input.transfer_kv_infos) {
    read_transfer_kv_info(buffer, transfer);
  }
  // eplb_info
  read_eplb_info(buffer, forward_input.eplb_info);

  // TODO: Optimize this logic. Placing this tensor directly on contiguous
  // device memory causes unknown errors. This needs to be optimized after the
  // root cause is identified and the error is resolved.
  read_tensor(buffer, input_params.new_cache_slots);
  read_tensor(buffer, input_params.block_tables);
}

inline void serialize_raw_forward_input(const RawForwardInput& input,
                                        char*& buffer) {
  write_vector_to_tensor(buffer, input.flatten_tokens_vec);
  if (input.flatten_positions_vec.size() > 0) {
    write_vector_to_tensor(buffer, input.flatten_positions_vec);
  } else {
    write_2d_vector_to_tensor(buffer, input.m_positions_vec);
  }

  // ModelInputParams
  write_data(buffer, input.empty_kv_cache);
  write_data(buffer, input.global_empty_kv_cache);
  write_data(buffer, input.batch_forward_type.value());
  write_data(buffer, input.num_sequences);
  write_data(buffer, input.max_seq_len);
  write_data(buffer, input.q_max_seq_len);
  write_data(buffer, input.batch_id);
  write_vector_to_tensor(buffer, input.q_seq_lens);
  write_vector_to_tensor(buffer, input.seq_lens);
  write_vector_to_tensor(buffer, input.paged_kv_indptr);
  write_vector_to_tensor(buffer, input.paged_kv_indices);
  write_vector_to_tensor(buffer, input.paged_kv_last_page_len);
  write_vector_to_tensor(buffer, input.new_cache_slot_offsets);
  write_vector_to_tensor(buffer, input.kv_cache_start_offsets);
  write_2d_vector_to_tensor(buffer, input.embeddings);
  write_vector(buffer, input.dp_global_token_nums);
  write_vector(buffer, input.dp_is_decode);
  write_vector(buffer, input.embedding_ids);
  write_vector(buffer, input.extra_token_ids);
  write_swap_blocks(buffer, input.swap_blocks);
  write_vector_to_tensor(buffer, input.src_block_indices);
  write_vector_to_tensor(buffer, input.dst_block_indices);
  write_vector_to_tensor(buffer, input.cum_sum);
  write_mm_batch_data(buffer, input.mm_data);
  write_vector_to_tensor(buffer, input.kv_cache_tokens_nums);

  // SamplingParameters
  write_data(buffer, input.selected_token_idxes.size());
  if (input.selected_token_idxes.size() > 0) {
    SamplingParameters sampling_params;
    sampling_params.init(input.sampling_params,
                         input.selected_token_idxes,
                         input.sample_idxes,
                         input.unique_token_ids_vec,
                         input.unique_token_counts_vec,
                         input.unique_token_lens_vec);

    write_tensor(buffer, sampling_params.selected_token_idxes);
    write_tensor(buffer, sampling_params.frequency_penalties);
    write_tensor(buffer, sampling_params.presence_penalties);
    write_tensor(buffer, sampling_params.repetition_penalties);
    write_tensor(buffer, sampling_params.temperatures);
    write_tensor(buffer, sampling_params.top_p);
    write_tensor(buffer, sampling_params.top_k);
    write_tensor(buffer, sampling_params.unique_token_ids);
    write_tensor(buffer, sampling_params.unique_token_counts);
    write_tensor(buffer, sampling_params.unique_token_ids_lens);
    write_tensor(buffer, sampling_params.sample_idxes);
    write_tensor(buffer, sampling_params.do_sample);
    write_data(buffer, sampling_params.all_random_sample);
    write_data(buffer, sampling_params.all_greedy_sample);
    write_data(buffer, sampling_params.logprobs);
    write_data(buffer, sampling_params.is_embeddings);
    write_data(buffer, sampling_params.max_top_logprobs);
    write_data(buffer, sampling_params.use_beam_search);
  }
  // acc_logprob
  write_vector_to_tensor(buffer, input.acc_logprob_vec);

  // transfer_kv_infos
  write_data(buffer, (uint64_t)input.transfer_kv_infos.size());
  for (const auto& t : input.transfer_kv_infos) {
    write_transfer_kv_info(buffer, t);
  }
  // eplb_info
  write_eplb_info(buffer, input.eplb_info);

  // TODO: Optimize this logic. Placing this tensor directly on contiguous
  // device memory causes unknown errors. This needs to be optimized after the
  // root cause is identified and the error is resolved.
  write_vector_to_tensor(buffer, input.new_token_slot_ids);
  write_2d_vector_to_tensor(buffer, input.block_tables_vec);
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

  read_vector_tensor(buffer, output.mm_embeddings);
}

void serialize_raw_forward_output(const RawForwardOutput& output,
                                  char*& buffer) {
  write_data(buffer, static_cast<uint64_t>(output.outputs.size()));
  for (const auto& sample : output.outputs) {
    write_raw_sample_output(buffer, sample);
  }

  write_vector(buffer, output.expert_load_data);

  write_data(buffer, output.prepared_layer_id);

  write_vector_tensor(buffer, output.mm_embeddings);
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
  input_params.kv_cache_tokens_nums =
      torch::tensor(std::move(raw_input.kv_cache_tokens_nums), tensor_options);

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

void convert_tensor_to_raw_output(
    const torch::Tensor& next_tokens,
    const torch::Tensor& logprobs,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    const torch::Tensor& embeddings,
    const std::vector<torch::Tensor>& mm_embeddings,
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
  raw_output.mm_embeddings = mm_embeddings;
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

}  // namespace

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

bool ForwardSharedMemoryManager::raw_input_write(const RawForwardInput& input) {
  uint64_t total_size = sizeof(ControlMetadata);
  total_size += type_size<uint64_t> + calculate_raw_forward_input_size(input);
  if (unlikely(total_size > size())) {
    LOG(ERROR) << "raw input size overflow, total_size: " << total_size
               << ", shm size: " << size();
    return false;
  }

  char* data_ptr = static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  write_data(data_ptr,
             total_size - sizeof(ControlMetadata) - type_size<uint64_t>);
  serialize_raw_forward_input(input, data_ptr);

  uint64_t real_size =
      (uint64_t)(data_ptr - static_cast<char*>(base_address()));
  CHECK(total_size == real_size) << "total_size != real_size.";
  std::atomic_thread_fence(std::memory_order_release);
  control_ptr_->version = ++last_version_;

  return true;
}

void ForwardSharedMemoryManager::raw_input_read(ForwardInput& input,
                                                const torch::Device& device) {
  while (true) {
    if (control_ptr_->version != last_version_) {
      last_version_ = control_ptr_->version;
      break;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(NUM_WAIT_NANOSECONDS));
  }

  const char* data_ptr =
      static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  uint64_t total_size;
  read_data(data_ptr, total_size);
  deserialize_raw_forward_input(data_ptr, total_size, input, device);

  return;
}

bool ForwardSharedMemoryManager::raw_output_write(
    const torch::Tensor& next_tokens,
    const torch::Tensor& logprobs,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    const torch::Tensor& embeddings,
    const std::vector<torch::Tensor>& mm_embeddings,
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
                               mm_embeddings,
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