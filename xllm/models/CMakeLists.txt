include(cc_library)

if(USE_NPU)
  include_directories(
    ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
  )
endif()

cc_library(
  NAME
    models
  HDRS
    model_registry.h
    models.h
  SRCS
    model_registry.cpp
  DEPS
    :model
    :dit_cache
    :model_loader
    :layers
    $<$<BOOL:${USE_NPU}>:npu_layers>
)
