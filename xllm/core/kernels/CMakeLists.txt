include(cc_library)

if(USE_NPU)
  add_subdirectory(npu)
endif()
