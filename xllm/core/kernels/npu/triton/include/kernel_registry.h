#ifndef KERNEL_REGISTRY_H
#define KERNEL_REGISTRY_H

#include <string>
#include <unordered_map>
#include <vector>

class KernelHandle;

class KernelRegistry {
 public:
  static KernelRegistry& getInstance();

  // 禁用拷贝和赋值
  KernelRegistry(const KernelRegistry&) = delete;
  KernelRegistry& operator=(const KernelRegistry&) = delete;

  // 注册kernel二进制文件
  bool registerKernel(const std::string& kernelName,
                      const std::string& binaryPath);

  // 批量注册kernels
  bool registerKernels(
      const std::vector<std::pair<std::string, std::string>>& kernelConfigs);

  // 获取kernel调用句柄
  KernelHandle getKernelHandle(const std::string& kernelName) const;

  // 检查kernel是否已注册
  bool isKernelRegistered(const std::string& kernelName) const;

  // 获取所有已注册的kernel名称
  std::vector<std::string> getRegisteredKernels() const;

  // 清理所有kernel
  void cleanup();

  // 解析 JSON 配置文件，提取 kernel_name, mix_mode, workspace_size,
  // lock_init_value, lock_num
  bool parseJsonConfig(const std::string& jsonPath,
                       std::string& kernelName,
                       std::string& mixMode,
                       int64_t& workspaceSize,
                       int64_t& lockInitValue,
                       int64_t& lockNum);

  // 获取 kernel 的 workspace 和 lock 配置
  bool getKernelWorkspaceConfig(const std::string& kernelName,
                                int64_t& workspaceSize,
                                int64_t& lockInitValue,
                                int64_t& lockNum) const;

 private:
  KernelRegistry() = default;
  ~KernelRegistry();

  struct KernelInfo {
    std::string name;
    char* buffer;
    void* stubFunc;  // 存储注册时使用的函数指针
    std::string persistentFuncName;
    std::string mixMode;    // 从 JSON 中读取的 mix_mode
    int64_t workspaceSize;  // workspace_size，默认 -1
    int64_t lockInitValue;  // lock_init_value，默认 -1
    int64_t lockNum;        // lock_num，默认 -1
  };

  char* loadBinaryFile(const std::string& filePath, uint32_t& fileSize);
  bool registerBinary(KernelInfo& info,
                      uint32_t binarySize,
                      void** stubFunc);  // 返回实际的stubFunc

  std::unordered_map<std::string, KernelInfo> kernelInfos_;  // 改为存储完整信息
};

#endif  // KERNEL_REGISTRY_H
