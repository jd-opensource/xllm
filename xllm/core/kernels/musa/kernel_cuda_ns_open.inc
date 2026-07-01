#if defined(XLLM_TORCH_MUSA)
#pragma push_macro("cuda")
#undef cuda
#endif
namespace xllm::kernel::cuda {
