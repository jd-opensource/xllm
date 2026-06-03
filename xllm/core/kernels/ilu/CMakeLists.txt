include(cc_library)
set(CMAKE_CUDA_ARCHITECTURES ivcore11)
file(GLOB_RECURSE ILU_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/*.cuh"
)

file(GLOB_RECURSE ILU_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.cu"
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

cc_library(
  NAME
    ilu_kernels
  HDRS
    ${ILU_HEADER_FILES}
  SRCS
    ${ILU_SOURCE_FILES}
  DEPS
    torch
    :util
    ixformer_kernels
    ixformer
    ${Python3_LIBRARIES}
    cuinfer
)

cc_test(
  NAME
    fused_qknorm_rope_test
  SRCS
    fused_qknorm_rope_test.cpp
  DEPS
    :ilu_kernels
    torch
    GTest::gtest_main
)

# ivcore clang++: *.cu needs -x ivcore; c10/torch headers need C++17 (_v traits,
# optional, string_view) when compiling the host portion of that translation unit.
if(TARGET ilu_kernels)
  get_target_property(_ilu_kernels_sources ilu_kernels SOURCES)
  if(_ilu_kernels_sources)
    foreach(_src IN LISTS _ilu_kernels_sources)
      if(_src MATCHES "\\.cu$")
        set_source_files_properties(
          "${_src}" PROPERTIES COMPILE_OPTIONS "-x;ivcore;-std=gnu++17")
      endif()
    endforeach()
  endif()
endif()
