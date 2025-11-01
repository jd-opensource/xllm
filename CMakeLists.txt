cmake_minimum_required(VERSION 3.26)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(USE_NPU "Enable NPU support" OFF)
option(USE_MLU "Enable MLU support" OFF)

if(DEVICE_ARCH STREQUAL "ARM")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(RUST_TARGET aarch64-unknown-linux-gnu)
endif()

if(USE_NPU)
  if(DEVICE_TYPE STREQUAL "USE_A3")
      add_compile_definitions(USE_A3) 
      add_definitions(-DUSE_A3)
      set(USE_A3 ON)
      message(STATUS "Building for device: A3 (macro USE_A3 defined)")
  else()
      add_compile_definitions(USE_A2)
      add_definitions(-DUSE_A2)
      set(USE_A2 ON)
      message(STATUS "Building for device: A2 (macro USE_A2 defined)")
  endif()

  option(INSTALL_XLLM_KERNELS "Install xllm_kernels RPM" ON)  
  message(STATUS "INSTALL_XLLM_KERNELS enabled: ${INSTALL_XLLM_KERNELS}")
  if(INSTALL_XLLM_KERNELS)
    if(DEVICE_TYPE STREQUAL "USE_A3")
        message("downloading a3 arm xllm kernels")
        file(DOWNLOAD 
            "https://9n-online-service.s3-internal.cn-north-1.jdcloud-oss.com/9n-xllm-atb/xllm_kernels-1.3.0-Linux.rpm"
            "${CMAKE_BINARY_DIR}/xllm_kernels.rpm"
        )
    else()  
      if(DEVICE_ARCH STREQUAL "ARM")
          message("downloading a2 arm xllm_kernels")
          file(DOWNLOAD 
              "https://9n-das-tools.s3.cn-north-1.jdcloud-oss.com/xllm-ai/xllm_kernels/0.6.0/xllm_kernels-1.3.1-Linux.a2.arm.rpm"
              "${CMAKE_BINARY_DIR}/xllm_kernels.rpm"
          )
      else()
          message("downloading a2 x86 xllm_kernels")
          file(DOWNLOAD 
              "https://9n-das-tools.s3.cn-north-1.jdcloud-oss.com/xllm-ai/xllm_kernels/0.6.0/xllm_kernels-1.3.1-Linux.a2.x86.rpm"
              "${CMAKE_BINARY_DIR}/xllm_kernels.rpm"
          )
      endif()
    endif()
      execute_process(COMMAND rpm -ivh  --replacepkgs --replacefiles "${CMAKE_BINARY_DIR}/xllm_kernels.rpm")
      file(WRITE "${CMAKE_BINARY_DIR}/.xllm_installed" "")
  endif()

  execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}/third_party/xllm_ops" rev-parse HEAD
    OUTPUT_VARIABLE XLLM_OPS_GIT_HEAD
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if(NOT DEFINED XLLM_OPS_GIT_HEAD_CACHED OR NOT XLLM_OPS_GIT_HEAD STREQUAL XLLM_OPS_GIT_HEAD_CACHED)
    message(STATUS "xllm_ops git HEAD changed; running precompile via execute_process")
    execute_process(
      COMMAND bash ${CMAKE_SOURCE_DIR}/third_party/xllm_ops/build.sh
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/xllm_ops
      RESULT_VARIABLE XLLM_OPS_RESULT
    )
    if(NOT XLLM_OPS_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to precompile xllm ops, error code: ${XLLM_OPS_RESULT}")
    endif()
    set(XLLM_OPS_GIT_HEAD_CACHED "${XLLM_OPS_GIT_HEAD}" CACHE INTERNAL "" FORCE)
    message(STATUS "xllm ops precompiled and HEAD cache updated")
  else()
    message(STATUS "xllm_ops git HEAD unchanged; skipping precompile")
  endif()
endif()

enable_testing()

if(NOT TARGET all_tests)
  add_custom_target(all_tests)
endif()

if(NOT TARGET export_module)
  add_custom_target(export_module)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

option(USE_MSPTI "Enable MSPTI for NPU Profiling" OFF)
if(USE_MSPTI)
  add_definitions(-DUSE_MSPTI)
endif()

option(USE_CCACHE "Attempt using CCache to wrap the compilation" ON)
# option(USE_CXX11_ABI "Use the new C++-11 ABI, which is not backwards compatible." OFF)
option(USE_MANYLINUX "Build for manylinux" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(USE_NPU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
elseif(USE_MLU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MESSAGE_LOG_LEVEL STATUS)
set(CMAKE_VERBOSE_MAKEFILE ON)

if(POLICY CMP0135)
  # CMP0135: ExternalProject ignores timestamps in archives by default for the URL download method.
  cmake_policy(SET CMP0135 NEW)
endif()

function(parse_make_options options prefix)
  foreach(option ${options})
    string(REGEX REPLACE "(-D|-)" "" option ${option})
    string(REPLACE "=" ";" option ${option})
    list(GET option 0 option_name)
    list(GET option 1 option_value)
    set(${prefix}_${option_name}
        ${option_value}
        PARENT_SCOPE)
  endforeach()
endfunction()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" 
      CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." 
      FORCE
  )
endif()

# Convert the bool variable to integer.
if(USE_CXX11_ABI)
  set(USE_CXX11_ABI 1)
  message(STATUS "Using the C++-11 ABI.")
else()
  set(USE_CXX11_ABI 0)
  message(STATUS "Using the pre C++-11 ABI.")
endif()

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
    if (DEFINED ENV{CCACHE_DIR})
      message(STATUS "Using CCACHE_DIR: $ENV{CCACHE_DIR}")
    endif()
  else()
    message(WARNING "Could not find ccache. Consider installing ccache to speed up compilation.")
  endif()
endif()

# if defined, create and use the default binary cache for vcpkg
if (DEFINED ENV{VCPKG_DEFAULT_BINARY_CACHE})
  file(MAKE_DIRECTORY $ENV{VCPKG_DEFAULT_BINARY_CACHE})
  message(STATUS "Using VCPKG_DEFAULT_BINARY_CACHE: $ENV{VCPKG_DEFAULT_BINARY_CACHE}")
endif()

if (DEFINED ENV{DEPENDENCES_ROOT})
  message(STATUS "Using DEPENDENCES_ROOT: $ENV{DEPENDENCES_ROOT}")
endif()

# configure vcpkg
# have to set CMAKE_TOOLCHAIN_FILE before first project call.
# if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
if (DEFINED ENV{VCPKG_ROOT})
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
  message(STATUS "VCPKG_ROOT found, using vcpkg at $ENV{VCPKG_ROOT}")
else()
  include(FetchContent)
  if (DEFINED ENV{DEPENDENCES_ROOT})
    set(VCPKG_SOURCE_DIR $ENV{DEPENDENCES_ROOT}/vcpkg-src)
  else()
    set(VCPKG_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/vcpkg-src)
  endif()

  if (USE_CXX11_ABI)
    FetchContent_Declare(vcpkg
      GIT_REPOSITORY "https://github.com/microsoft/vcpkg.git"
      GIT_TAG "2024.02.14"
      SOURCE_DIR ${VCPKG_SOURCE_DIR}
    )
  else()
    FetchContent_Declare(vcpkg
      GIT_REPOSITORY "https://gitcode.com/xLLM-AI/vcpkg.git"
      GIT_TAG "ffc42e97c866ce9692f5c441394832b86548422c" #disable cxx11_abi
      SOURCE_DIR ${VCPKG_SOURCE_DIR}
    )
    message(STATUS "Using custom vcpkg with cxx11_abi disabled")
  endif()
  FetchContent_MakeAvailable(vcpkg)

  message(STATUS "Downloading and using vcpkg at ${vcpkg_SOURCE_DIR}")
  set(CMAKE_TOOLCHAIN_FILE ${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake
      CACHE STRING "Vcpkg toolchain file")
endif()

set(CPPREST_EXCLUDE_WEBSOCKETS ON CACHE BOOL "Exclude websockets functionality." FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-truncation")

project("xllm" LANGUAGES C CXX)

# find_package(CUDAToolkit REQUIRED)

# setup CMake module path, defines path for include() and find_package()
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
enable_language(Rust)
find_package(Rust REQUIRED)

if(UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
endif()

find_package(Boost REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(Threads REQUIRED)
# find all dependencies from vcpkg
find_package(fmt CONFIG REQUIRED GLOBAL)
find_package(glog CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(leveldb CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(re2 CONFIG REQUIRED)
find_package(folly CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(Python COMPONENTS Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

if (USE_MANYLINUX)
  # manylinux doesn't ship Development.Embed
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
else()
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
endif()

if (USE_CXX11_ABI)
  # only use jemalloc if using the new C++-11 ABI
  find_package(Jemalloc)
  if(Jemalloc_FOUND)
    link_libraries(Jemalloc::jemalloc)
  endif()
endif()

# Important Note: Always invoke find_package for other dependencies
# before including libtorch, as doing so afterwards may lead to
# unexpected linker errors.
if (DEFINED ENV{LIBTORCH_ROOT})
  find_package(Torch REQUIRED HINTS "$ENV{LIBTORCH_ROOT}")
  message(STATUS "Using libtorch at $ENV{LIBTORCH_ROOT}")
else()
  include(FetchContent)
  if (USE_CXX11_ABI)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip")
  else()
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.1.0%2Bcpu.zip")
  endif()

  if (DEFINED ENV{DEPENDENCES_ROOT})
    set(LIBTORCH_SOURCE_DIR $ENV{DEPENDENCES_ROOT}/libtorch-src)
  else()
    set(LIBTORCH_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/libtorch-src)
  endif()

  FetchContent_Declare(libtorch 
    URL ${LIBTORCH_URL} 
    SOURCE_DIR ${LIBTORCH_SOURCE_DIR}
  )
  FetchContent_MakeAvailable(libtorch)
  
  find_package(Torch REQUIRED PATHS ${LIBTORCH_SOURCE_DIR} NO_DEFAULT_PATH)
  message(STATUS "Downloading and using libtorch 2.1.0 for CPU at ${LIBTORCH_SOURCE_DIR}")
endif()

if(USE_NPU)
  add_definitions(-DUSE_NPU)
  add_definitions(-DBUILD_LIBTORCH)
  add_definitions(-DTORCH_SETCUSTOMHANDLER=ON)
  # set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
  set(CMAKE_VERBOSE_MAKEFILE ON)
  add_definitions(-DTORCH_HIGHER_THAN_PTA6)
  include_directories(
      $ENV{PYTHON_INCLUDE_PATH}
      $ENV{PYTORCH_INSTALL_PATH}/include
      $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
      $ENV{PYTORCH_NPU_INSTALL_PATH}/include
      $ENV{NPU_HOME_PATH}/include
      $ENV{ATB_HOME_PATH}/include
      $ENV{NPU_HOME_PATH}/opp/vendors/xllm/op_api/include/
      $ENV{XLLM_KERNELS_PATH}/include/xllm_kernels/core/include
      $ENV{XLLM_KERNELS_PATH}/include/xllm_kernels
      $ENV{XLLM_KERNELS_PATH}/include/
  )

  link_directories(
      $ENV{PYTHON_LIB_PATH}
      $ENV{PYTORCH_INSTALL_PATH}/lib
      $ENV{PYTORCH_NPU_INSTALL_PATH}/lib
      $ENV{NPU_HOME_PATH}/lib64
      $ENV{ATB_HOME_PATH}/lib
      $ENV{NPU_TOOLKIT_HOME}/lib64
      $ENV{NPU_HOME_PATH}/opp/vendors/xllm/op_api/lib/
      $ENV{XLLM_KERNELS_PATH}/lib/
  )
  link_libraries(cust_opapi)
endif()

if(USE_MLU)
  add_definitions(-DUSE_MLU)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  include_directories(
      $ENV{PYTHON_INCLUDE_PATH}
      $ENV{PYTORCH_INSTALL_PATH}/include
      $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
      $ENV{PYTORCH_MLU_INSTALL_PATH}
      $ENV{PYTORCH_MLU_INSTALL_PATH}/../
      $ENV{PYTORCH_MLU_INSTALL_PATH}/csrc
      $ENV{NEUWARE_HOME}/include
  )

  link_directories(
    $ENV{PYTHON_LIB_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/lib
    $ENV{PYTORCH_MLU_INSTALL_PATH}/csrc/lib
    $ENV{PYTORCH_MLU_INSTALL_PATH}
    $ENV{NEUWARE_HOME}/lib64
  )
endif()

# check if USE_CXX11_ABI is set correctly
# if (DEFINED USE_CXX11_ABI)
#   parse_make_options(${TORCH_CXX_FLAGS} "TORCH_CXX_FLAGS")
#   if(DEFINED TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI
#      AND NOT ${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} EQUAL ${USE_CXX11_ABI})
#       message(FATAL_ERROR
#           "The libtorch compilation options _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} "
#           "found by CMake conflict with the project setting USE_CXX11_ABI=${USE_CXX11_ABI}.")
#   endif()
# endif()

# carry over torch flags to the rest of the project
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
add_compile_options(${TORCH_CXX_FLAGS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DC10_USE_GLOG")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

# enable testing in this directory so we can do a top-level `make test`.
# this also includes the BUILD_TESTING option, which is on by default.
include(CTest)
include(GoogleTest)
option(BUILD_TESTING "Build the testing tree." ON)

# include current path
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/xllm/core)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/etcd_cpp_apiv3)

# brpc
set(BRPC_OUTPUT_DIR ${CMAKE_BINARY_DIR}/third_party/brpc/output)
include_directories(${BRPC_OUTPUT_DIR}/include)
link_directories(${BRPC_OUTPUT_DIR}/lib)

if(USE_NPU)
  # hccl_transfer
  include_directories(third_party/hccl_transfer/hccl_transfer/include)
endif()

# add subdirectories
add_subdirectory(xllm)
add_subdirectory(third_party)
