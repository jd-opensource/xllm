// 定义 xllm 使用triton_kernel的接口
#include "npu_triton_add_kernel.h"

namespace TritonNPU {

void validateTensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.defined(), name, "tensor is not defined");
  TORCH_CHECK(t.is_contiguous(), name, "tensor must be contiguous");
  TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1);
}

torch::Tensor npu_triton_add_kernel(const torch::Tensor& x,
                                    const torch::Tensor& y,
                                    int64_t nElements,
                                    int32_t gridX,
                                    int32_t gridY,
                                    int32_t gridZ) {
  validateTensor(x, "input_x");
  validateTensor(y, "input_y");
  TORCH_CHECK(x.sizes() == y.sizes(), "input sizes must be same.");
  TORCH_CHECK(gridX > 0, "gridX (BLOCK_SIZE) must > 0.");

  const int64_t maxElements = x.numel();
  torch::Tensor out = torch::empty_like(x);

  int32_t device_id = x.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  void* xPtr = const_cast<void*>(x.data_ptr());
  void* yPtr = const_cast<void*>(y.data_ptr());
  void* outPtr = out.data_ptr();

  auto ret = TritonLauncher::add_kernel(stream,
                                        gridX,
                                        gridY,
                                        gridZ,
                                        nullptr,
                                        nullptr,
                                        xPtr,
                                        yPtr,
                                        outPtr,
                                        static_cast<int32_t>(nElements));
  TORCH_CHECK(
      ret == RT_ERROR_NONE, "launch_add_kernel failed with error ", ret);
  return out;
}

}  // namespace TritonNPU
