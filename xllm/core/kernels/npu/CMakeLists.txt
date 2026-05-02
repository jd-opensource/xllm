include(cc_library)

add_subdirectory(tilelang)

add_subdirectory(xllm_ops)

cc_library(
  NAME
    npu_kernels
  HDRS
    npu_ops_api.h
    aclnn/pytorch_npu_helper.hpp
  SRCS
    active.cpp
    attention.cpp
    fused_layernorm.cpp
    matmul.cpp
    npu_causal_conv1d.cpp
    npu_gemma_rms_norm.cpp
    npu_grouped_matmul.cpp
    npu_moe_gating_topk_softmax.cpp
    npu_moe_init_routing_v2.cpp
    npu_moe_token_unpermute.cpp
    rope.cpp
    npu_recurrent_gated_delta_rule.cpp
    rec_constrained_topk.cpp
  DEPS
    :torch_npu_kernels
    :tilelang_kernels
)
