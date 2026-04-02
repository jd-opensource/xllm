cmake_minimum_required(VERSION 3.10)

project(xllm_capi_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Dependencies — Protobuf (vcpkg uses protobuf-config.cmake; package name is "protobuf")
# -----------------------------------------------------------------------------
# Recommended (matches main project / vcpkg.json):
#   cmake -B build \
#     -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
#     -DVCPKG_MANIFEST_DIR=<path/to/xllm/repo_root>
# To reuse an existing xLLM build without re-installing deps:
#   -DCMAKE_PREFIX_PATH=<xllm_build>/vcpkg_installed/<triplet>
#
# Fallback: system FindProtobuf / pkg-config / manual paths (no vcpkg).
find_package(protobuf CONFIG QUIET)
if(NOT protobuf_FOUND)
  find_package(Protobuf CONFIG QUIET)
endif()

# vcpkg protobuf::protoc may not set Protobuf_PROTOC_EXECUTABLE; resolve from target.
if((protobuf_FOUND OR Protobuf_FOUND) AND TARGET protobuf::libprotobuf)
  if(NOT Protobuf_PROTOC_EXECUTABLE AND TARGET protobuf::protoc)
    get_target_property(Protobuf_PROTOC_EXECUTABLE protobuf::protoc IMPORTED_LOCATION)
    if(NOT Protobuf_PROTOC_EXECUTABLE)
      get_target_property(Protobuf_PROTOC_EXECUTABLE protobuf::protoc IMPORTED_LOCATION_RELEASE)
    endif()
    if(NOT Protobuf_PROTOC_EXECUTABLE)
      get_target_property(Protobuf_PROTOC_EXECUTABLE protobuf::protoc IMPORTED_LOCATION_DEBUG)
    endif()
  endif()
  if(NOT Protobuf_PROTOC_EXECUTABLE)
    message(FATAL_ERROR
      "Protobuf CONFIG found but could not resolve protoc (protobuf::protoc / Protobuf_PROTOC_EXECUTABLE).")
  endif()
  message(STATUS "Found Protobuf (CONFIG / vcpkg): protoc=${Protobuf_PROTOC_EXECUTABLE}")
  set(_PROTOBUF_LIB protobuf::libprotobuf)
else()
  set(Protobuf_FOUND FALSE)
  set(protobuf_FOUND FALSE)
  find_package(Protobuf QUIET)

  if(Protobuf_FOUND)
    message(STATUS "Found Protobuf (module): protoc=${Protobuf_PROTOC_EXECUTABLE}")
  endif()

  if(NOT Protobuf_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(PC_PROTOBUF QUIET protobuf)
    endif()

    if(PC_PROTOBUF_FOUND)
      set(Protobuf_INCLUDE_DIRS ${PC_PROTOBUF_INCLUDE_DIRS})
      set(Protobuf_LIBRARIES ${PC_PROTOBUF_LIBRARIES})
      find_program(Protobuf_PROTOC_EXECUTABLE protoc)
      if(NOT Protobuf_PROTOC_EXECUTABLE)
        message(FATAL_ERROR
          "Found libprotobuf via pkg-config, but protoc was not found in PATH. "
          "Please install protoc or set Protobuf_PROTOC_EXECUTABLE.")
      endif()
      set(_PROTOBUF_LIB ${Protobuf_LIBRARIES})
    endif()
  endif()

  if(NOT Protobuf_FOUND AND NOT PC_PROTOBUF_FOUND)
    find_path(Protobuf_INCLUDE_DIR
      NAMES google/protobuf/message.h
      PATHS /usr/include /usr/local/include
    )
    find_library(Protobuf_LIBRARY
      NAMES protobuf libprotobuf
      PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64
    )
    find_program(Protobuf_PROTOC_EXECUTABLE
      NAMES protoc
      PATHS /usr/bin /usr/local/bin
    )

    if(Protobuf_INCLUDE_DIR AND Protobuf_LIBRARY AND Protobuf_PROTOC_EXECUTABLE)
      set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIR})
      set(Protobuf_LIBRARIES ${Protobuf_LIBRARY})
      set(_PROTOBUF_LIB ${Protobuf_LIBRARY})
      message(STATUS "Found Protobuf manually:"
                     " include=${Protobuf_INCLUDE_DIR}"
                     " lib=${Protobuf_LIBRARY}"
                     " protoc=${Protobuf_PROTOC_EXECUTABLE}")
    else()
      message(FATAL_ERROR
        "Could NOT find Protobuf.\n"
        "Preferred: configure with vcpkg like the main xLLM build, e.g.\n"
        "  -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake\n"
        "  -DVCPKG_MANIFEST_DIR=<xllm_repo_root>\n"
        "Or install system protobuf + protoc, or set:\n"
        "  -DProtobuf_INCLUDE_DIR=/path/to/include\n"
        "  -DProtobuf_LIBRARY=/path/to/libprotobuf.so\n"
        "  -DProtobuf_PROTOC_EXECUTABLE=/path/to/protoc\n")
    endif()
  endif()

  if(TARGET protobuf::libprotobuf)
    set(_PROTOBUF_LIB protobuf::libprotobuf)
  endif()
endif()

# -----------------------------------------------------------------------------
# Generate protobuf sources from xllm_test.proto
# -----------------------------------------------------------------------------
set(XLLM_TEST_PROTO ${CMAKE_CURRENT_LIST_DIR}/xllm_test.proto)
set(XLLM_TEST_PB_CC ${CMAKE_CURRENT_BINARY_DIR}/xllm_test.pb.cc)
set(XLLM_TEST_PB_H ${CMAKE_CURRENT_BINARY_DIR}/xllm_test.pb.h)

add_custom_command(
  OUTPUT ${XLLM_TEST_PB_CC} ${XLLM_TEST_PB_H}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
          --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
          -I ${CMAKE_CURRENT_LIST_DIR}
          ${XLLM_TEST_PROTO}
  DEPENDS ${XLLM_TEST_PROTO}
  COMMENT "Generating xllm_test.pb.cc/h from xllm_test.proto"
  VERBATIM
)

add_library(c_api_test_proto STATIC ${XLLM_TEST_PB_CC})
target_include_directories(c_api_test_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
if(TARGET protobuf::libprotobuf)
  target_include_directories(c_api_test_proto PUBLIC
    $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>)
elseif(Protobuf_INCLUDE_DIRS)
  target_include_directories(c_api_test_proto PUBLIC ${Protobuf_INCLUDE_DIRS})
endif()
# PRIVATE: xllm_test controls link order (brpc before protobuf) for static archives.
target_link_libraries(c_api_test_proto PRIVATE ${_PROTOBUF_LIB})

# -----------------------------------------------------------------------------
# brpc + gflags + glog (server binary; brpc uses glog — link glog after libbrpc.a)
# -----------------------------------------------------------------------------
find_package(gflags CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

# brpc: main xLLM may use <repo>/build/third_party/brpc/output or
# <repo>/build/<toolchain>/third_party/brpc/output (e.g. cmake.linux-aarch64-*).
# Headers: output/include next to the library, else third_party/brpc/src.
get_filename_component(_XLLM_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.."
                       ABSOLUTE)
if(NOT BRPC_ROOT)
  if(DEFINED ENV{BRPC_ROOT} AND NOT "$ENV{BRPC_ROOT}" STREQUAL "")
    set(BRPC_ROOT "$ENV{BRPC_ROOT}")
  else()
    set(BRPC_ROOT
        "${_XLLM_REPO_ROOT}/build/third_party/brpc/output")
  endif()
endif()
set(BRPC_ROOT "${BRPC_ROOT}" CACHE PATH
    "brpc prefix: include/brpc/server.h and lib/libbrpc.a (override if needed)")

set(BRPC_LIB "")
foreach(_brpc_lib_cand
    "${BRPC_ROOT}/lib/libbrpc.a"
    "${BRPC_ROOT}/lib/libbrpc.so"
    "${_XLLM_REPO_ROOT}/build/third_party/brpc/output/lib/libbrpc.a"
    "${_XLLM_REPO_ROOT}/build/third_party/brpc/output/lib/libbrpc.so")
  if(EXISTS "${_brpc_lib_cand}")
    set(BRPC_LIB "${_brpc_lib_cand}")
    break()
  endif()
endforeach()
# e.g. build/cmake.linux-aarch64-cpython-311/third_party/brpc/output/lib/...
if(NOT BRPC_LIB)
  file(GLOB _brpc_lib_glob LIST_DIRECTORIES false
    "${_XLLM_REPO_ROOT}/build/*/third_party/brpc/output/lib/libbrpc.a"
    "${_XLLM_REPO_ROOT}/build/*/third_party/brpc/output/lib/libbrpc.so")
  if(_brpc_lib_glob)
    list(SORT _brpc_lib_glob)
    list(GET _brpc_lib_glob 0 BRPC_LIB)
  endif()
endif()
if(NOT BRPC_LIB)
  find_library(BRPC_LIB NAMES brpc libbrpc.a
    PATHS "${BRPC_ROOT}/lib"
          "${CMAKE_PREFIX_PATH}/lib"
          /usr/local/lib
          /usr/lib64
          /usr/lib)
endif()

set(BRPC_INCLUDE_DIR "")
if(BRPC_LIB)
  get_filename_component(_brpc_out "${BRPC_LIB}" DIRECTORY)
  get_filename_component(_brpc_out "${_brpc_out}" DIRECTORY)
  if(EXISTS "${_brpc_out}/include/brpc/server.h")
    set(BRPC_INCLUDE_DIR "${_brpc_out}/include")
  endif()
endif()
if(NOT BRPC_INCLUDE_DIR AND EXISTS "${BRPC_ROOT}/include/brpc/server.h")
  set(BRPC_INCLUDE_DIR "${BRPC_ROOT}/include")
endif()
if(NOT BRPC_INCLUDE_DIR
   AND EXISTS "${_XLLM_REPO_ROOT}/third_party/brpc/src/brpc/server.h")
  set(BRPC_INCLUDE_DIR "${_XLLM_REPO_ROOT}/third_party/brpc/src")
endif()

if(NOT BRPC_INCLUDE_DIR)
  message(FATAL_ERROR
    "brpc headers not found (BRPC_ROOT='${BRPC_ROOT}').\n"
    "  Expected include/brpc/server.h next to libbrpc, or submodule\n"
    "  '${_XLLM_REPO_ROOT}/third_party/brpc/src/brpc/server.h'.\n"
    "  Set -DBRPC_ROOT=... to a prefix with include/brpc/server.h, or init\n"
    "  git submodules so third_party/brpc exists.")
endif()
if(NOT BRPC_LIB)
  message(FATAL_ERROR
    "libbrpc not found (BRPC_ROOT='${BRPC_ROOT}').\n"
    "  xllm_test links brpc from the main xLLM build, e.g.\n"
    "    ${_XLLM_REPO_ROOT}/build/third_party/brpc/output/lib/libbrpc.a\n"
    "    ${_XLLM_REPO_ROOT}/build/*/third_party/brpc/output/lib/libbrpc.a\n"
    "  Build the main project first, or pass -DBRPC_ROOT=/path/to/.../output\n"
    "  (directory with include/ and lib/libbrpc.a).")
endif()

find_package(leveldb CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

# -----------------------------------------------------------------------------
# Build xllm_test (links external /usr/local/xllm/lib/libxllm.so)
# -----------------------------------------------------------------------------
add_executable(xllm_test
  ${CMAKE_CURRENT_LIST_DIR}/xllm_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
)

# rec.h / types.h: installed C API under /usr/local/xllm/include (run xllm/c_api/install.sh).
# Local test headers (utils.h): ${CMAKE_CURRENT_LIST_DIR}
target_include_directories(xllm_test
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    /usr/local/xllm/include
    ${CMAKE_CURRENT_LIST_DIR}
    ${BRPC_INCLUDE_DIR}
)
target_link_directories(xllm_test PRIVATE /usr/local/xllm/lib)
# Static libbrpc.a pulls protobuf gzip + glog symbols; libprotobuf must appear
# after brpc (or use --start-group) so GzipOutputStream etc. resolve; ZLIB for gzip.
# --start-group/--end-group: GNU ld needs them so libprotobuf resolves symbols
# referenced from libbrpc.a (e.g. GzipOutputStream) when linking static archives.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT MSVC)
  set(_BRPC_LINK_GROUP_START -Wl,--start-group)
  set(_BRPC_LINK_GROUP_END -Wl,--end-group)
else()
  set(_BRPC_LINK_GROUP_START "")
  set(_BRPC_LINK_GROUP_END "")
endif()
target_link_libraries(xllm_test
  PRIVATE
    c_api_test_proto
    gflags::gflags
    ${_BRPC_LINK_GROUP_START}
    ${BRPC_LIB}
    ${_PROTOBUF_LIB}
    glog::glog
    ZLIB::ZLIB
    ${_BRPC_LINK_GROUP_END}
    leveldb::leveldb
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
    dl
    xllm
)

# Keep runtime able to locate libxllm.so without setting LD_LIBRARY_PATH.
set_target_properties(xllm_test PROPERTIES
  BUILD_RPATH "/usr/local/xllm/lib"
  INSTALL_RPATH "/usr/local/xllm/lib"
)
