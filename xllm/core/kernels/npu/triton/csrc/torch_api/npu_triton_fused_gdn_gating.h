#ifndef NPU_TRITON_FUSED_GDN_GATING_H
#define NPU_TRITON_FUSED_GDN_GATING_H

#include <torch/torch.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

namespace triton {
namespace npu {

std::pair<torch::Tensor, torch::Tensor> npu_fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float beta = 1.0f,
    float threshold = 20.0f);

}  // namespace npu
}  // namespace triton
#endif
