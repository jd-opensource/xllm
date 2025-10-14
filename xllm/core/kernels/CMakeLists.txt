include(cc_library)

if(USE_NPU)
  include_directories(
    ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
  )
  add_subdirectory(npu)
endif()

if(USE_MLU)
  add_subdirectory(mlu)
endif()
