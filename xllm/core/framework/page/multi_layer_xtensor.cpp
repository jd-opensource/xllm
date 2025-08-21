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

#include "multi_layer_xtensor.h"

namespace xllm {

MultiLayerXTensor::MultiLayerXTensor(
    std::vector<std::shared_ptr<XTensor>>& xtensors)
    : xtensors_(xtensors) {
  num_layers_ = xtensors_.size();
  int32_t max_seqs_per_batch = xtensors_[0]->options().max_seqs_per_batch();

  phy_page_ids_vec_.resize(max_seqs_per_batch);
  num_free_seq_ids_ = max_seqs_per_batch;
  free_seq_ids_.reserve(max_seqs_per_batch);
  for (int32_t i = max_seqs_per_batch - 1; i >= 0; i--) {
    free_seq_ids_.push_back(i);
  }
}

void MultiLayerXTensor::append_phy_pages(
    int32_t seq_id,
    const std::vector<uint32_t>& new_phy_pages) {
  phy_page_ids_vec_[seq_id].insert(phy_page_ids_vec_[seq_id].end(),
                                   new_phy_pages.begin(),
                                   new_phy_pages.end());
}

void MultiLayerXTensor::free(int32_t seq_id) {
  for (size_t layer_idx = 0; layer_idx < num_layers_; layer_idx++) {
    VirPtr vir_ptr = get_vir_ptr(seq_id, layer_idx);
#if defined(USE_NPU)
    VmmResult status = aclrtUnmapMem(vir_ptr);
    CHECK_EQ(status, VmmSuccess) << "Failed to unmap virtual memory for layer "
                                 << layer_idx << " of sequence " << seq_id;
#endif
  }
  deallocate_seq_id(seq_id);
}

int32_t MultiLayerXTensor::allocate_seq_id() {
  CHECK_GT(num_free_seq_ids_, 0) << "No more available seq_id!";
  return free_seq_ids_[--num_free_seq_ids_];
}

void MultiLayerXTensor::deallocate_seq_id(int32_t seq_id) {
  CHECK_LT(num_free_seq_ids_, free_seq_ids_.size());
  free_seq_ids_[num_free_seq_ids_++] = seq_id;
}

}  // namespace xllm