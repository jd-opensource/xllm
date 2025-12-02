#include "kernel_loader.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "kernel_registry.h"

KernelHandle KernelLoader::loadKernel(const std::string& kernelName,
                                      const std::string& binaryPath) {
  auto& registry = KernelRegistry::getInstance();
  if (registry.registerKernel(kernelName, binaryPath)) {
    return registry.getKernelHandle(kernelName);
  }
  return KernelHandle();
}

std::vector<KernelHandle> KernelLoader::loadKernels(
    const std::vector<std::pair<std::string, std::string>>& kernelConfigs) {
  auto& registry = KernelRegistry::getInstance();
  std::vector<KernelHandle> handles;

  if (registry.registerKernels(kernelConfigs)) {
    for (const auto& config : kernelConfigs) {
      handles.push_back(registry.getKernelHandle(config.first));
    }
  }

  return handles;
}

KernelHandle KernelLoader::getKernel(const std::string& kernelName) {
  auto& registry = KernelRegistry::getInstance();
  return registry.getKernelHandle(kernelName);
}

bool KernelLoader::getKernelWorkspaceConfig(const std::string& kernelName,
                                            int64_t& workspaceSize,
                                            int64_t& lockInitValue,
                                            int64_t& lockNum) {
  auto& registry = KernelRegistry::getInstance();
  return registry.getKernelWorkspaceConfig(
      kernelName, workspaceSize, lockInitValue, lockNum);
}

int KernelLoader::loadKernelsFromDirectory(const std::string& directoryPath) {
  if (!std::filesystem::exists(directoryPath)) {
    std::cerr << "Error: Directory does not exist: " << directoryPath
              << std::endl;
    return -1;
  }

  if (!std::filesystem::is_directory(directoryPath)) {
    std::cerr << "Error: Path is not a directory: " << directoryPath
              << std::endl;
    return -1;
  }

  std::cout << "Info: Scanning directory for kernel binaries: " << directoryPath
            << std::endl;

  // 扫描目录，只支持 .npubin 格式
  std::vector<std::pair<std::string, std::string>> kernelConfigs;

  for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    std::string filePath = entry.path().string();
    std::string fileName = entry.path().filename().string();
    std::string extension = entry.path().extension().string();

    // 只支持 .npubin 格式
    if (extension != ".npubin") {
      continue;
    }

    // 尝试从 JSON 文件中读取 kernel_name
    std::filesystem::path jsonPathObj = entry.path();
    jsonPathObj.replace_extension(".json");
    std::string jsonPath = jsonPathObj.string();

    // 默认使用文件名（去掉扩展名）作为 kernel_name
    size_t lastDot = fileName.find_last_of('.');
    std::string kernelName =
        (lastDot != std::string::npos) ? fileName.substr(0, lastDot) : fileName;

    // 如果 JSON 文件存在，尝试读取 kernel_name
    if (std::filesystem::exists(jsonPath)) {
      try {
        std::ifstream jsonFile(jsonPath);
        if (jsonFile.is_open()) {
          nlohmann::json j;
          jsonFile >> j;
          jsonFile.close();

          if (j.contains("kernel_name") && j["kernel_name"].is_string()) {
            kernelName = j["kernel_name"].get<std::string>();
            std::cout << "Info: Found kernel '" << kernelName
                      << "' from JSON: " << jsonPath << std::endl;
          } else {
            std::cout
                << "Info: JSON file exists but no kernel_name, using filename: "
                << kernelName << std::endl;
          }
        }
      } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to parse JSON file " << jsonPath
                  << ", using filename as kernel name: " << e.what()
                  << std::endl;
      }
    } else {
      std::cout << "Info: No JSON file found for " << fileName
                << ", using filename as kernel name: " << kernelName
                << std::endl;
    }

    kernelConfigs.push_back({kernelName, filePath});
  }

  if (kernelConfigs.empty()) {
    std::cout << "Info: No kernel binary files found in directory: "
              << directoryPath << std::endl;
    return 0;
  }

  std::cout << "Info: Found " << kernelConfigs.size()
            << " kernel binary file(s)" << std::endl;

  // 批量注册
  auto& registry = KernelRegistry::getInstance();
  if (registry.registerKernels(kernelConfigs)) {
    std::cout << "Info: Successfully registered " << kernelConfigs.size()
              << " kernel(s)" << std::endl;
    return static_cast<int>(kernelConfigs.size());
  } else {
    std::cerr << "Error: Failed to register some kernels" << std::endl;
    return -1;
  }
}

void KernelLoader::cleanup() {
  auto& registry = KernelRegistry::getInstance();
  registry.cleanup();
}
