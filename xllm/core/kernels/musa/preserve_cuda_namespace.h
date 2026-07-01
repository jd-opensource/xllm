#pragma once

// custom_defines.h (via mcc_wrapper) maps token cuda->musa for ATen/cuda headers.
// Include this immediately before xllm::kernel::cuda definitions in .cu files.
#ifdef cuda
#undef cuda
#endif
