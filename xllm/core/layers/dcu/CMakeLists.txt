#
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

include(cc_library)

# libflash_attention - pre-built shared library for prefix prefill/decode.
#
# FLASH_ATTENTION_LIB: directory containing libflash_attention.so.
# No headers are needed. The caller (flash_attention.cpp) provides its own
# forward declarations for prefix_prefill_varlen_fwd / prefix_decode_varlen_fwd.
#
# Configure via:
#   - Environment variable: export FLASH_ATTENTION_LIB=/path/to/dir
#   - CMake argument:       cmake -DFLASH_ATTENTION_LIB=/path/to/dir ...
if(NOT TARGET flash_attention)
  set(FLASH_ATTENTION_LIB
      "$ENV{FLASH_ATTENTION_LIB}"
      CACHE PATH "Directory containing libflash_attention.so")

  if(NOT FLASH_ATTENTION_LIB)
    message(FATAL_ERROR
      "FLASH_ATTENTION_LIB is not set.\n"
      "  Option 1: export FLASH_ATTENTION_LIB=/path/to/dir\n"
      "  Option 2: cmake -DFLASH_ATTENTION_LIB=/path/to/dir ...\n"
      "The directory should contain libflash_attention.so.")
  endif()

  find_library(
    _flash_attn_lib_path flash_attention
    PATHS "${FLASH_ATTENTION_LIB}"
    NO_DEFAULT_PATH)

  if(_flash_attn_lib_path)
    add_library(flash_attention SHARED IMPORTED)
    set_target_properties(
      flash_attention PROPERTIES
      IMPORTED_LOCATION "${_flash_attn_lib_path}"
      INTERFACE_COMPILE_DEFINITIONS "BUILD_FA_KVCACHE")
    message(STATUS "Found libflash_attention: ${_flash_attn_lib_path}")
  else()
    message(FATAL_ERROR
      "libflash_attention.so not found in ${FLASH_ATTENTION_LIB}.")
  endif()
endif()

cc_library(
  NAME
    dcu_layers
  HDRS
    attention.h
    base_attention_impl.h
    torch_attention.h
    flash_attention.h
  SRCS
    attention.cpp
    base_attention_impl.cpp
    torch_attention.cpp
    flash_attention.cpp
  DEPS
    :common_layers
    flash_attention
    dcu_kernels
)
