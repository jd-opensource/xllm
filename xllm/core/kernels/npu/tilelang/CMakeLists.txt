include(cc_library)
include(cc_test)
include(CMakeParseArguments)

if(NOT USE_NPU)
  return()
endif()

if(NOT Python_EXECUTABLE)
  message(FATAL_ERROR "Python_EXECUTABLE is required for TileLang codegen")
endif()

find_program(BISHENG_EXECUTABLE bisheng REQUIRED)
if(NOT BISHENG_EXECUTABLE)
  message(FATAL_ERROR "bisheng not found in PATH")
endif()

if(NOT DEFINED ENV{NPU_HOME_PATH})
  message(FATAL_ERROR "NPU_HOME_PATH is not set")
endif()

if(NOT DEFINED ENV{TL_ROOT})
  message(FATAL_ERROR "TL_ROOT is not set")
endif()

set(TILELANG_PY_DIR "${CMAKE_SOURCE_DIR}/xllm/core/kernels/python/tilelang")
set(TILELANG_BISHENG_ARCH "dav-c220")

set(TILELANG_BISHENG_COMMON_FLAGS
  -O2
  -std=gnu++17
  -xcce
  -fPIC
  -mllvm -cce-aicore-stack-size=0x8000
  -mllvm -cce-aicore-function-stack-size=0x8000
  -mllvm -cce-aicore-record-overflow=true
  -mllvm -cce-aicore-addr-transform
  -mllvm -cce-aicore-dcci-insert-for-scalar=false
  -DL2_CACHE_HINT
  -DBACKEND_HYBM
)

set(TILELANG_BISHENG_INCLUDE_DIRS
  "$ENV{NPU_HOME_PATH}/include"
  "$ENV{NPU_HOME_PATH}/include/experiment/runtime"
  "$ENV{NPU_HOME_PATH}/include/experiment/msprof"
  "$ENV{NPU_HOME_PATH}/compiler/tikcpp"
  "$ENV{NPU_HOME_PATH}/compiler/tikcpp/tikcfw"
  "$ENV{NPU_HOME_PATH}/compiler/tikcpp/tikcfw/impl"
  "$ENV{NPU_HOME_PATH}/compiler/tikcpp/tikcfw/interface"
  "$ENV{TL_ROOT}/3rdparty/catlass/include"
  "$ENV{TL_ROOT}/3rdparty/shmem/include"
  "$ENV{TL_ROOT}/3rdparty/shmem/src/device"
  "$ENV{TL_ROOT}/src"
)

set(TILELANG_BISHENG_INCLUDE_FLAGS "")
foreach(_inc IN LISTS TILELANG_BISHENG_INCLUDE_DIRS)
  list(APPEND TILELANG_BISHENG_INCLUDE_FLAGS "-I${_inc}")
endforeach()

function(tilelang_add_ascendc_kernel)
  set(options "")
  set(oneValueArgs TARGET PY_SCRIPT BUILD_DIR KERNEL_NAME BISHENG_ARCH SOURCE_ENTRY_SYMBOL)
  set(multiValueArgs CODEGEN_ARGS DEPENDS)
  cmake_parse_arguments(TL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(_required TARGET PY_SCRIPT BUILD_DIR KERNEL_NAME)
    if(NOT TL_${_required})
      message(FATAL_ERROR "tilelang_add_ascendc_kernel: missing ${_required}")
    endif()
  endforeach()

  if(NOT TL_BISHENG_ARCH)
    set(TL_BISHENG_ARCH "${TILELANG_BISHENG_ARCH}")
  endif()

  if(NOT TL_SOURCE_ENTRY_SYMBOL)
    set(TL_SOURCE_ENTRY_SYMBOL "call")
  endif()

  file(MAKE_DIRECTORY "${TL_BUILD_DIR}")

  set(_kernel_cpp "${TL_BUILD_DIR}/${TL_TARGET}_kernel.cpp")
  set(_kernel_obj "${TL_BUILD_DIR}/${TL_TARGET}_kernel.o")
  set(_entry_symbol "${TL_KERNEL_NAME}_call")

  add_custom_command(
    OUTPUT "${_kernel_cpp}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${TL_BUILD_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E env
            "${Python_EXECUTABLE}" "${TL_PY_SCRIPT}"
            --output "${_kernel_cpp}"
            ${TL_CODEGEN_ARGS}
    COMMAND sed -i -e "s/\\<${TL_SOURCE_ENTRY_SYMBOL}\\>/${_entry_symbol}/g" "${_kernel_cpp}"
    DEPENDS "${TL_PY_SCRIPT}" ${TL_DEPENDS}
    COMMENT "TileLang codegen: ${_kernel_cpp}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${_kernel_obj}"
    COMMAND "${BISHENG_EXECUTABLE}"
            "--cce-aicore-arch=${TL_BISHENG_ARCH}"
            ${TILELANG_BISHENG_COMMON_FLAGS}
            ${TILELANG_BISHENG_INCLUDE_FLAGS}
            "${_kernel_cpp}"
            -c
            -o "${_kernel_obj}"
    DEPENDS "${_kernel_cpp}"
    COMMENT "bisheng compile: ${_kernel_obj}"
    VERBATIM
  )

  set_source_files_properties("${_kernel_obj}" PROPERTIES
    GENERATED TRUE
    EXTERNAL_OBJECT TRUE
  )
  add_custom_target(${TL_TARGET}_obj_gen DEPENDS "${_kernel_obj}")

  set(${TL_TARGET}_KERNEL_OBJ "${_kernel_obj}" PARENT_SCOPE)
  set(${TL_TARGET}_ENTRY_SYMBOL "${_entry_symbol}" PARENT_SCOPE)
  set(${TL_TARGET}_OBJ_TARGET "${TL_TARGET}_obj_gen" PARENT_SCOPE)
endfunction()

# ---------------------------
# RoPE kernel configuration
# ---------------------------
set(TILELANG_ROPE_HEAD_DIM 576)
set(TILELANG_ROPE_ROPE_DIM 64)
set(TILELANG_ROPE_KERNEL_NAME "rope_in_place")
set(TILELANG_ROPE_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/rope")

tilelang_add_ascendc_kernel(
  TARGET tilelang_rope
  PY_SCRIPT "${TILELANG_PY_DIR}/rope.py"
  BUILD_DIR "${TILELANG_ROPE_BUILD_DIR}"
  KERNEL_NAME "${TILELANG_ROPE_KERNEL_NAME}"
  CODEGEN_ARGS
    --head-dim "${TILELANG_ROPE_HEAD_DIM}"
    --rope-dim "${TILELANG_ROPE_ROPE_DIM}"
)

# Add more TileLang kernels:
# 1) call tilelang_add_ascendc_kernel(...)
# 2) append *_KERNEL_OBJ / *_OBJ_TARGET to the lists below.
set(TILELANG_KERNEL_OBJECTS)
set(TILELANG_KERNEL_DEP_TARGETS)
list(APPEND TILELANG_KERNEL_OBJECTS "${tilelang_rope_KERNEL_OBJ}")
list(APPEND TILELANG_KERNEL_DEP_TARGETS "${tilelang_rope_OBJ_TARGET}")

cc_library(
  NAME
    tilelang_kernels
  HDRS
    rope_wrapper.h
  SRCS
    rope_wrapper.cpp
    ${TILELANG_KERNEL_OBJECTS}
  DEPS
    torch
    torch_npu
)
add_dependencies(tilelang_kernels ${TILELANG_KERNEL_DEP_TARGETS})

target_compile_definitions(tilelang_kernels PRIVATE
  XLLM_TL_ROPE_HEAD_DIM=${TILELANG_ROPE_HEAD_DIM}
  XLLM_TL_ROPE_ROPE_DIM=${TILELANG_ROPE_ROPE_DIM}
  XLLM_TL_ROPE_ENTRY=${tilelang_rope_ENTRY_SYMBOL}
)

target_link_libraries(tilelang_kernels
  PUBLIC
    "$ENV{NPU_HOME_PATH}/lib64/libascendcl.so"
    "$ENV{NPU_HOME_PATH}/lib64/libruntime.so"
    "$ENV{NPU_HOME_PATH}/lib64/libplatform.so"
    "$ENV{NPU_HOME_PATH}/lib64/libc_sec.so"
)

cc_test(
  NAME
    rope_wrapper_test
  SRCS
    rope_wrapper_test.cpp
  DEPS
    :tilelang_kernels
    torch
    GTest::gtest_main
    glog::glog
)

target_compile_definitions(rope_wrapper_test PRIVATE
  XLLM_TL_ROPE_HEAD_DIM=${TILELANG_ROPE_HEAD_DIM}
  XLLM_TL_ROPE_ROPE_DIM=${TILELANG_ROPE_ROPE_DIM}
)
