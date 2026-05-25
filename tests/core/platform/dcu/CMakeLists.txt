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
include(cc_test)

cc_test(
  NAME
    vmm_api_dcu_test
  SRCS
    vmm_api_dcu_test.cpp
  DEPS
    :platform
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(vmm_api_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
set_target_properties(vmm_api_dcu_test PROPERTIES LINKER_LANGUAGE HIP)

cc_test(
  NAME
    stream_dcu_test
  SRCS
    stream_dcu_test.cpp
  DEPS
    :platform
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(stream_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
set_target_properties(stream_dcu_test PROPERTIES LINKER_LANGUAGE HIP)

cc_test(
  NAME
    device_dcu_test
  SRCS
    device_dcu_test.cpp
  DEPS
    :platform
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(device_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
set_target_properties(device_dcu_test PROPERTIES LINKER_LANGUAGE HIP)

cc_test(
  NAME
    shared_vmm_allocator_dcu_test
  SRCS
    shared_vmm_allocator_dcu_test.cpp
  DEPS
    :platform
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(shared_vmm_allocator_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
set_target_properties(shared_vmm_allocator_dcu_test PROPERTIES LINKER_LANGUAGE HIP)

cc_test(
  NAME
    numa_utils_dcu_test
  SRCS
    numa_utils_dcu_test.cpp
  DEPS
    :platform
    GTest::gtest_main
    glog::glog
)
target_compile_definitions(numa_utils_dcu_test PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
set_target_properties(numa_utils_dcu_test PROPERTIES LINKER_LANGUAGE HIP)
