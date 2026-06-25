set(TRITON_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KERNEL_OUTPUT_DIR ${TRITON_KERNEL_DIR}/kernel)
set(KERNEL_LIB_DIR ${KERNEL_OUTPUT_DIR}/lib)
set(KERNEL_INCLUDE_DIR ${KERNEL_OUTPUT_DIR}/include)

if(NOT MLU_CNRT_LIBRARY)
  find_library(MLU_CNRT_LIBRARY
    NAMES cnrt
    PATHS "$ENV{NEUWARE_HOME}/lib64"
  )
  if(NOT MLU_CNRT_LIBRARY)
    message(FATAL_ERROR "Failed to find MLU libcnrt.so in $ENV{NEUWARE_HOME}/lib64")
  endif()
endif()

file(MAKE_DIRECTORY ${KERNEL_LIB_DIR})
file(MAKE_DIRECTORY ${KERNEL_INCLUDE_DIR})

execute_process(
    COMMAND python3 ${TRITON_KERNEL_DIR}/compile_kernel.py
            --output-dir ${KERNEL_OUTPUT_DIR}
            --generate-cmake-only
    WORKING_DIRECTORY ${TRITON_KERNEL_DIR}
    RESULT_VARIABLE GEN_CMAKE_RESULT
    OUTPUT_VARIABLE GEN_CMAKE_OUTPUT
    ERROR_VARIABLE GEN_CMAKE_ERROR
)

if(NOT GEN_CMAKE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate CMake config: ${GEN_CMAKE_ERROR}")
endif()

include(${KERNEL_LIB_DIR}/kernel_objects.cmake)
add_custom_command(
    OUTPUT
        ${TRITON_KERNEL_OBJECT_FILES}
        ${TRITON_KERNEL_GEN_C_FILES}
    COMMAND python3 ${TRITON_KERNEL_DIR}/compile_kernel.py
            --output-dir ${KERNEL_OUTPUT_DIR}
            --archs mtp_592 mtp_613
    WORKING_DIRECTORY ${TRITON_KERNEL_DIR}
    COMMENT "Compiling Triton Kernel for MLU backend"
    VERBATIM
)

# Make custom target depend on all generated files
add_custom_target(triton_kernel_compile ALL
    DEPENDS
        ${TRITON_KERNEL_OBJECT_FILES}
        ${TRITON_KERNEL_GEN_C_FILES}
)

set(TRITON_KERNEL_OBJECT_FILES ${TRITON_KERNEL_OBJECT_FILES} PARENT_SCOPE)
set(TRITON_KERNEL_INCLUDE_DIR ${KERNEL_INCLUDE_DIR} PARENT_SCOPE)

# Create object library for generated .c files
add_library(tmo_kernels_objects OBJECT ${TRITON_KERNEL_GEN_C_FILES})
add_dependencies(tmo_kernels_objects triton_kernel_compile)

# Create static library from object files
add_library(tmo_kernels STATIC)
target_link_libraries(tmo_kernels PRIVATE tmo_kernels_objects)
target_link_libraries(tmo_kernels PRIVATE
  ${TRITON_KERNEL_OBJECT_FILES}
  -Wl,--no-as-needed
  ${MLU_CNRT_LIBRARY}
  -Wl,--as-needed
)
target_include_directories(tmo_kernels PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
  ${KERNEL_INCLUDE_DIR}
)

# Link dependencies
target_link_libraries(tmo_kernels PRIVATE
  torch
  :platform
  cnclep
  torch_mlu
  -Wl,--no-as-needed
  torch_mlu_ops
  ${MLU_CNRT_LIBRARY}
  -Wl,--as-needed
  Python::Python
)


add_library(:tmo_kernels ALIAS tmo_kernels)
