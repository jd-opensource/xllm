#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <string>
#include <vector>

#include "kernel_handle.h"

class KernelLoader {
 public:
  // 加载单个kernel
  static KernelHandle loadKernel(const std::string& kernelName,
                                 const std::string& binaryPath);

  // 批量加载kernels
  static std::vector<KernelHandle> loadKernels(
      const std::vector<std::pair<std::string, std::string>>& kernelConfigs);

  // 从指定目录加载所有算子二进制
  // 会自动扫描目录中的 .npubin 文件，并尝试从对应的 JSON 文件中读取 kernel_name
  // 如果 JSON 不存在或没有 kernel_name，则使用文件名（去掉扩展名）作为
  // kernel_name 返回成功注册的 kernel 数量，失败返回 -1
  static int loadKernelsFromDirectory(const std::string& directoryPath);

  // 获取kernel句柄（已加载的情况下）
  static KernelHandle getKernel(const std::string& kernelName);

  // 获取kernel的workspace和lock配置
  static bool getKernelWorkspaceConfig(const std::string& kernelName,
                                       int64_t& workspaceSize,
                                       int64_t& lockInitValue,
                                       int64_t& lockNum);

  // 清理所有加载的kernels
  static void cleanup();

 private:
  KernelLoader() = delete;
  ~KernelLoader() = delete;
};

#endif  // KERNEL_LOADER_H
