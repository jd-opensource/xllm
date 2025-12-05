include(cc_library)
include(cc_test)

# Define the base dependencies
set(BASE_DEPS
  :common
  :flags
  :layers
  :prefix_cache
  :block
  :processors
  :chat_template
  glog::glog
  torch
  torch_python
)

if(USE_NPU)
  list(APPEND BASE_DEPS :platform_npu)
endif()

# Define the library
cc_library(
  NAME 
    model
  HDRS
    causal_lm.h
    causal_vlm.h
    dit_model.h
    embedding_lm.h
    embedding_vlm.h
    mm_embedding_vlm.h
    model_args.h
    npu_dp_ep_padding.h
    model_input_params.h
  SRCS
    npu_dp_ep_padding.cpp
  DEPS
    ${BASE_DEPS}
)
target_link_libraries(model PRIVATE :kv_cache)

if(USE_NPU)
cc_test(
    NAME 
      npu_dp_ep_padding_test
    SRCS
      npu_dp_ep_padding_test.cpp
    DEPS
      :flags
      :parallel_state
      torch
      model
      absl::synchronization
      absl::time
      GTest::gtest_main
)

target_link_libraries(npu_dp_ep_padding_test
                 PUBLIC Python::Python
                 $<$<BOOL:${USE_NPU}>:ascendcl>
                 $<$<BOOL:${USE_NPU}>:hccl>
                 $<$<BOOL:${USE_NPU}>:c_sec>
                 $<$<BOOL:${USE_NPU}>:nnopbase>)
endif()
