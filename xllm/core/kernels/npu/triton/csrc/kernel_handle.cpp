#include "kernel_handle.h"

KernelHandle::KernelHandle(const std::string& kernelName, const char* handle)
    : kernelName_(kernelName), handle_(handle) {}
