#pragma once

#undef check_tensor_contiguous
#define check_tensor_contiguous(x, type) \
  TORCH_CHECK(x.scalar_type() == type);  \
  TORCH_CHECK(x.is_cuda());              \
  TORCH_CHECK(x.is_contiguous());

#undef check_tensor_half_bf_float
#define check_tensor_half_bf_float(x)                       \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half ||    \
              x.scalar_type() == at::ScalarType::Float ||   \
              x.scalar_type() == at::ScalarType::BFloat16); \
  TORCH_CHECK(x.is_cuda());

// from torchCheckMsgImpl
inline const char* ixformerCheckMsgImpl(const char* msg) { return msg; }
// // If there is just 1 user-provided C-string argument, use it.
// inline const char *ixformerCheckMsgImpl(const char *msg, const char *args) {
//     return args;
// }

#define IXFORMER_CHECK_MSG(cond, type, ...)                                   \
  (ixformerCheckMsgImpl("Expected " #cond " to be true, but got false.  "     \
                        "(Could this error message be improved?  If so, "     \
                        "please report an enhancement request to ixformer.)", \
                        ##__VA_ARGS__))

#define IXFORMER_CHECK(cond, ...)                                            \
  {                                                                          \
    if (!(cond)) {                                                           \
      std::cerr << __FILE__ << " (" << __LINE__ << ")"                       \
                << "-" << __FUNCTION__ << " : "                              \
                << IXFORMER_CHECK_MSG(cond, "", ##__VA_ARGS__) << std::endl; \
      throw std::runtime_error("IXFORMER_CHECK ERROR");                      \
    }                                                                        \
  }

#undef CUINFER_CHECK
#define CUINFER_CHECK(func)                                                \
  do {                                                                     \
    cuinferStatus_t status = (func);                                       \
    if (status != CUINFER_STATUS_SUCCESS) {                                \
      std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ \
                << ": " << cuinferGetErrorString(status) << std::endl;     \
      throw std::runtime_error("CUINFER_CHECK ERROR");                     \
    }                                                                      \
  } while (0)
