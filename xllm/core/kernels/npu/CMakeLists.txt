cmake_minimum_required(VERSION 3.26)
project(xLLM)

include_directories(
    ${CMAKE_SOURCE_DIR}/xllm/core/kernels/npu
    ${CMAKE_SOURCE_DIR}/third_party/
    ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
    $ENV{ATB_HOME_PATH}/include
    $ENV{NPU_HOME_PATH}/include
)

link_directories(
    $ENV{NPU_HOME_PATH}/lib64
    $ENV{ATB_HOME_PATH}/lib
    $ENV{NPU_TOOLKIT_HOME}/lib64)

add_subdirectory(xllm_ops)