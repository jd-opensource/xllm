#include <acl/acl.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include "../npu_triton_add_kernel.h"
#include "kernel_loader.h"

const int64_t SIZE = 98432;
const int32_t BLOCK_SIZE = 1024;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << "<kernel_bin_path> <kernel_name>"
              << std::endl;
    std::cout << "Example: " << argv[0]
              << "./binary/add_kernel.npu_bin add kernel" << std::endl;
  }

  std::string binaryPath = argv[1];
  std::string kernelName = argv[2];

  std::cout << "Binary:" << binaryPath << std::endl;
  std::cout << "Kernel:" << kernelName << std::endl;

  aclInit(nullptr);
  int32_t deviceId = 0;
  aclError ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) {
    std::cout << "Failed to set deivce: error = " << ret << std::endl;
  }

  // load kernel
  auto kernelHandle = KernelLoader::loadKernel(kernelName, binaryPath);
  if (!kernelHandle.isValid()) {
    std::cout << "Failed to load kernel: " << kernelName << std::endl;
    return -1;
  }

  aclrtStream stream;
  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) {
    std::cout << "Failede to create Stream, error = " << ret << std::endl;
    return -1;
  }

  torch::manual_seed(0);

  torch::Tensor x = torch::rand(
      {SIZE}, torch::TensorOptions().dtype(torch::kFloat32).device("npu:0"));
  torch::Tensor y = torch::rand(
      {SIZE}, torch::TensorOptions().dtype(torch::kFloat32).device("npu:0"));

  torch::Tensor output_torch = x + y;
  torch::Tensor output_triton =
      TritonNPU::npu_triton_add_kernel(x, y, SIZE, BLOCK_SIZE);

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    std::cout << "Stream Synchronized Failed: error = " << ret << std::endl;
    return -1;
  }

  torch::Tensor diff = torch::abs(output_torch - output_triton);
  float max_diff = torch::max(diff).item<float>();
  const float tolerance = 1e-5f;
  bool passed = max_diff < tolerance;

  if (passed) {
    std::cout << "test passed" << std::endl;
  } else {
    std::cout << "test failed " << std::endl;
  }
  ret = aclrtDestroyStream(stream);
  if (ret != ACL_SUCCESS) {
    std::cout << "aclDestroyStream Failed: error = " << ret << std::endl;
    return -1;
  }

  ret = aclFinalize();
  if (ret != ACL_SUCCESS) {
    std::cout << "aclFinalize Failed: error = " << ret << std::endl;
    return -1;
  }
  return 0;
}
