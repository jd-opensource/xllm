/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace xllm::npu {

// Abstract adapter between the model owning the vision encoder layers and the
// ACL graph infrastructure in core/runtime.
//
// The graph manager captures and replays the encoder loop purely through this
// interface, so it never depends on any concrete model type. A VLM that wants
// its encoder captured as an ACL graph implements this adapter and registers
// itself with VisionEncoderAclGraphManager::set_adapter().
//
// The forward signatures mirror the per-layer forward of a Qwen3-style vision
// transformer (flash-attention encoder block + deepstack patch mergers), but
// the interface itself is model-agnostic.
class VisionEncoderGraphAdapter {
 public:
  virtual ~VisionEncoderGraphAdapter() = default;

  // Number of encoder blocks in the captured loop.
  virtual int32_t num_encoder_layers() const = 0;

  // Layer indexes at which a deepstack patch merger output must be captured.
  // The returned vector is indexed by "deepstack slot"; slot k is emitted when
  // the loop reaches layer deepstack_indexes()[k].
  virtual const std::vector<int64_t>& deepstack_indexes() const = 0;

  // Run encoder block `layer_idx` and return the new hidden states.
  // `hidden` is the persistent input buffer owned by the graph manager; the
  // adapter must not alias it beyond this call.
  //
  // cos_pos/sin_pos/cu_seqlens/cu_seqlens_vec are taken by non-const reference
  // to match the existing vision encoder layer forward API
  // (NpuQwen3VisionEncoderLayer), which takes them as non-const ref.
  virtual torch::Tensor forward_encoder_layer(
      int32_t layer_idx,
      torch::Tensor& hidden,
      torch::Tensor& cos_pos,
      torch::Tensor& sin_pos,
      torch::Tensor& cu_seqlens,
      std::vector<int>& cu_seqlens_vec) = 0;

  // Run the deepstack patch merger at `deepstack_slot` on `hidden` and return
  // its output. `deepstack_slot` is the index into deepstack_indexes().
  virtual torch::Tensor forward_deepstack_merger(
      int32_t deepstack_slot,
      const torch::Tensor& hidden) = 0;
};

}  // namespace xllm::npu
