#include "kernel_registry.h"

#include <acl/acl.h>
#include <experiment/runtime/runtime/rt.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "kernel_handle.h"

KernelRegistry& KernelRegistry::getInstance() {
  static KernelRegistry instance;
  return instance;
}

KernelRegistry::~KernelRegistry() { cleanup(); }

bool KernelRegistry::parseJsonConfig(const std::string& jsonPath,
                                     std::string& kernelName,
                                     std::string& mixMode,
                                     int64_t& workspaceSize,
                                     int64_t& lockInitValue,
                                     int64_t& lockNum) {
  try {
    std::ifstream file(jsonPath);
    if (!file.is_open()) {
      std::cerr << "Warning: Cannot open JSON file: " << jsonPath << std::endl;
      return false;
    }

    nlohmann::json j;
    file >> j;
    file.close();

    // 读取 kernel_name
    if (j.contains("kernel_name") && j["kernel_name"].is_string()) {
      kernelName = j["kernel_name"].get<std::string>();
    } else {
      std::cerr << "Warning: JSON file missing 'kernel_name' field: "
                << jsonPath << std::endl;
      return false;
    }

    // 读取 mix_mode
    if (j.contains("mix_mode") && j["mix_mode"].is_string()) {
      mixMode = j["mix_mode"].get<std::string>();
    } else {
      std::cerr << "Warning: JSON file missing 'mix_mode' field, using default "
                   "'aiv': "
                << jsonPath << std::endl;
      mixMode = "aiv";  // 默认值
    }

    // 读取 workspace_size，如果不存在则默认为 -1
    if (j.contains("workspace_size")) {
      if (j["workspace_size"].is_number_integer()) {
        workspaceSize = j["workspace_size"].get<int64_t>();
      } else if (j["workspace_size"].is_null()) {
        workspaceSize = -1;
      } else {
        std::cerr
            << "Warning: 'workspace_size' is not an integer, using default -1"
            << std::endl;
        workspaceSize = -1;
      }
    } else {
      workspaceSize = -1;
    }

    // 读取 lock_init_value，如果不存在则默认为 -1
    if (j.contains("lock_init_value")) {
      if (j["lock_init_value"].is_number_integer()) {
        lockInitValue = j["lock_init_value"].get<int64_t>();
      } else if (j["lock_init_value"].is_null()) {
        lockInitValue = -1;
      } else {
        std::cerr
            << "Warning: 'lock_init_value' is not an integer, using default -1"
            << std::endl;
        lockInitValue = -1;
      }
    } else {
      lockInitValue = -1;
    }

    // 读取 lock_num，如果不存在则默认为 -1
    if (j.contains("lock_num")) {
      if (j["lock_num"].is_number_integer()) {
        lockNum = j["lock_num"].get<int64_t>();
      } else if (j["lock_num"].is_null()) {
        lockNum = -1;
      } else {
        std::cerr << "Warning: 'lock_num' is not an integer, using default -1"
                  << std::endl;
        lockNum = -1;
      }
    } else {
      lockNum = -1;
    }

    return true;
  } catch (const nlohmann::json::parse_error& e) {
    std::cerr << "Error: JSON parse error in " << jsonPath << ": " << e.what()
              << std::endl;
    return false;
  } catch (const std::exception& e) {
    std::cerr << "Error: Failed to parse JSON file " << jsonPath << ": "
              << e.what() << std::endl;
    return false;
  }
}

bool KernelRegistry::registerKernel(const std::string& kernelName,
                                    const std::string& binaryPath) {
  if (kernelInfos_.find(kernelName) != kernelInfos_.end()) {
    std::cout << "Info: Kernel '" << kernelName << "' is already registered"
              << std::endl;
    return true;  // 已注册
  }

  // 尝试读取 JSON 配置文件
  // JSON 文件路径：将 .o 或 .npubin 替换为 .json
  std::string jsonPath = binaryPath;
  size_t lastDot = jsonPath.find_last_of('.');
  if (lastDot != std::string::npos) {
    jsonPath = jsonPath.substr(0, lastDot) + ".json";
  } else {
    jsonPath = binaryPath + ".json";
  }

  std::string parsedKernelName = kernelName;
  std::string mixMode = "aiv";  // 默认值
  int64_t workspaceSize = -1;
  int64_t lockInitValue = -1;
  int64_t lockNum = -1;

  // 解析 JSON 配置（如果文件存在）
  if (std::filesystem::exists(jsonPath)) {
    if (parseJsonConfig(jsonPath,
                        parsedKernelName,
                        mixMode,
                        workspaceSize,
                        lockInitValue,
                        lockNum)) {
      std::cout << "Info: Parsed JSON config: kernel_name=" << parsedKernelName
                << ", mix_mode=" << mixMode
                << ", workspace_size=" << workspaceSize
                << ", lock_init_value=" << lockInitValue
                << ", lock_num=" << lockNum << std::endl;
    } else {
      std::cerr << "Warning: Failed to parse JSON config, using provided "
                   "kernel name and default values"
                << std::endl;
    }
  } else {
    std::cerr << "Warning: JSON config file not found: " << jsonPath
              << ", using provided kernel name and default values" << std::endl;
  }

  // 加载二进制文件
  uint32_t fileSize = 0;
  char* buffer = loadBinaryFile(binaryPath, fileSize);
  if (!buffer) {
    std::cerr << "Error: Failed to load binary file for kernel '" << kernelName
              << "'" << std::endl;
    return false;
  }

  // 创建kernel信息
  KernelInfo info;
  info.name = kernelName;  // 使用注册时提供的名称作为 key
  info.buffer = buffer;
  info.persistentFuncName =
      parsedKernelName;  // 使用 JSON 中解析出的 kernel_name
  info.mixMode = mixMode;
  info.workspaceSize = workspaceSize;
  info.lockInitValue = lockInitValue;
  info.lockNum = lockNum;

  // 注册
  void* stubFunc = nullptr;
  if (!registerBinary(info, fileSize, &stubFunc)) {
    std::cerr << "Error: Failed to register binary for kernel '" << kernelName
              << "'" << std::endl;
    delete[] buffer;
    return false;
  }

  info.stubFunc = stubFunc;
  kernelInfos_[kernelName] = info;

  std::cout << "Info: Successfully registered kernel '" << kernelName
            << "' (function: " << parsedKernelName << ")" << std::endl;
  return true;
}

bool KernelRegistry::registerKernels(
    const std::vector<std::pair<std::string, std::string>>& kernelConfigs) {
  bool allSuccess = true;
  size_t successCount = 0;
  for (const auto& config : kernelConfigs) {
    if (registerKernel(config.first, config.second)) {
      successCount++;
    } else {
      allSuccess = false;
    }
  }
  std::cout << "Info: Registered " << successCount << " out of "
            << kernelConfigs.size() << " kernel(s)." << std::endl;
  return allSuccess;
}

KernelHandle KernelRegistry::getKernelHandle(
    const std::string& kernelName) const {
  auto it = kernelInfos_.find(kernelName);
  if (it != kernelInfos_.end()) {
    // 使用注册时存储的stubFunc，而不是重新创建
    return KernelHandle(kernelName,
                        static_cast<const char*>(it->second.stubFunc));
  }
  return KernelHandle();
}

bool KernelRegistry::isKernelRegistered(const std::string& kernelName) const {
  return kernelInfos_.find(kernelName) != kernelInfos_.end();
}

std::vector<std::string> KernelRegistry::getRegisteredKernels() const {
  std::vector<std::string> kernels;
  for (const auto& pair : kernelInfos_) {
    kernels.push_back(pair.first);
  }
  return kernels;
}

bool KernelRegistry::getKernelWorkspaceConfig(const std::string& kernelName,
                                              int64_t& workspaceSize,
                                              int64_t& lockInitValue,
                                              int64_t& lockNum) const {
  auto it = kernelInfos_.find(kernelName);
  if (it != kernelInfos_.end()) {
    workspaceSize = it->second.workspaceSize;
    lockInitValue = it->second.lockInitValue;
    lockNum = it->second.lockNum;
    return true;
  }
  return false;
}

char* KernelRegistry::loadBinaryFile(const std::string& filePath,
                                     uint32_t& fileSize) {
  std::ifstream file(filePath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open binary file: " << filePath << std::endl;
    return nullptr;
  }

  fileSize = static_cast<uint32_t>(file.tellg());
  if (fileSize == 0) {
    std::cerr << "Error: Binary file is empty: " << filePath << std::endl;
    file.close();
    return nullptr;
  }

  file.seekg(0, std::ios::beg);

  char* buffer = new (std::nothrow) char[fileSize];
  if (!buffer) {
    std::cerr << "Error: Failed to allocate memory for binary file: "
              << filePath << " (size: " << fileSize << " bytes)" << std::endl;
    file.close();
    return nullptr;
  }

  file.read(buffer, fileSize);
  if (file.gcount() != static_cast<std::streamsize>(fileSize)) {
    std::cerr << "Error: Failed to read complete binary file: " << filePath
              << std::endl;
    delete[] buffer;
    file.close();
    return nullptr;
  }

  file.close();
  return buffer;
}

bool KernelRegistry::registerBinary(KernelInfo& info,
                                    uint32_t binarySize,
                                    void** stubFunc) {
  rtDevBinary_t binary;
  void* binHandle = nullptr;

  binary.data = info.buffer;
  binary.length = binarySize;
  // 根据 mix_mode 设置正确的 magic 值
  if (info.mixMode == "aiv") {
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  } else {  // TODO 支持 "aic"
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
  }
  binary.version = 0;

  rtError_t rtRet = rtDevBinaryRegister(&binary, &binHandle);
  if (rtRet != RT_ERROR_NONE) {
    std::cerr << "Error: rtDevBinaryRegister failed for kernel '" << info.name
              << "': error=" << rtRet << std::endl;
    return false;
  }

  // 使用持久化的函数名
  rtRet = rtFunctionRegister(binHandle,
                             info.persistentFuncName.c_str(),
                             info.persistentFuncName.c_str(),
                             (void*)info.persistentFuncName.c_str(),
                             0);

  if (rtRet != RT_ERROR_NONE) {
    std::cerr << "Error: rtFunctionRegister failed for kernel '" << info.name
              << "' with function name '" << info.persistentFuncName
              << "': error=" << rtRet << std::endl;
    // 注意：binHandle 由运行时管理，不需要手动释放
    return false;
  }

  *stubFunc = (void*)info.persistentFuncName.c_str();
  return true;
}

void KernelRegistry::cleanup() {
  size_t count = kernelInfos_.size();
  for (auto& [name, info] : kernelInfos_) {
    if (info.buffer) {
      delete[] info.buffer;
      info.buffer = nullptr;
    }
  }
  kernelInfos_.clear();
  std::cout << "Info: KernelRegistry cleanup completed. Unregistered " << count
            << " kernel(s)." << std::endl;
}
