#ifndef KERNEL_HANDLE_H
#define KERNEL_HANDLE_H

#include <string>

class KernelHandle {
 public:
  KernelHandle() = default;
  KernelHandle(const std::string& kernelName, const char* handle);

  // 直接用于 rtKernelLaunch 的接口
  const char* get() const { return handle_; }
  operator const char*() const { return handle_; }

  bool isValid() const { return handle_ != nullptr; }
  const std::string& getName() const { return kernelName_; }

 private:
  std::string kernelName_;
  const char* handle_{nullptr};
};

#endif  // KERNEL_HANDLE_H
