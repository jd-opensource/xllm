#include <torch/torch.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <limits>

#include "experiment/runtime/runtime/rt.h"
#include "kernel_launchers.h"
namespace TritonNPU {

torch::Tensor npu_triton_add_kernel(const torch::Tensor& x,
                                    const torch::Tensor& y,
                                    int64_t nElements,
                                    int32_t gridX,
                                    int32_t gridY = 1,
                                    int32_t gridZ = 1);

}  // namespace TritonNPU
