cmake_minimum_required(VERSION 3.26)

project(llm_examples_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)


link_directories(/usr/local/xllm/lib)

### build single_llm_instance
add_executable(single_llm_instance single_llm_instance.cpp service_request.cpp)
target_include_directories(single_llm_instance
    PRIVATE /usr/local/xllm/include
)

target_link_libraries(single_llm_instance
    PRIVATE xllm
)

### get cxx11 ABI config from torch api.
set(CHECK_TORCH_ABI_SCRIPT
"
import torch
print(0 if not torch.compiled_with_cxx11_abi() else 1)
"
)

execute_process(
    COMMAND python3 -c "${CHECK_TORCH_ABI_SCRIPT}"
    OUTPUT_VARIABLE TORCH_CXX11_ABI_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE 
    RESULT_VARIABLE TORCH_ABI_CHECK_RESULT
)

if(NOT TORCH_ABI_CHECK_RESULT EQUAL 0)
    message(WARNING "Failed to detect PyTorch CXX11 ABI! "
            "Reason: ${TORCH_ABI_CHECK_RESULT}\n"
            "Fallback to _GLIBCXX_USE_CXX11_ABI=0 (default)")
    set(TORCH_CXX11_ABI_VALUE 0)
else()
    message(STATUS "Detected PyTorch CXX11 ABI: ${TORCH_CXX11_ABI_VALUE} "
            "(0=OFF, 1=ON)")
endif()

if(TORCH_CXX11_ABI_VALUE EQUAL 0)
target_compile_definitions(single_llm_instance
    PRIVATE 
        _GLIBCXX_USE_CXX11_ABI=0 
        USE_CXX11_ABI=OFF
)
else()
target_compile_definitions(single_llm_instance
    PRIVATE 
        _GLIBCXX_USE_CXX11_ABI=1 
        USE_CXX11_ABI=ON
)
endif()


set_target_properties(single_llm_instance PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "/usr/local/xllm/lib"
    RPATH "/usr/local/xllm/lib"
)

### build multiple_llm_instances
add_executable(multiple_llm_instances multiple_llm_instances.cpp service_request.cpp)
target_include_directories(multiple_llm_instances
    PRIVATE /usr/local/xllm/include
)

target_link_libraries(multiple_llm_instances
    PRIVATE xllm
)

if(TORCH_CXX11_ABI_VALUE EQUAL 0)
target_compile_definitions(multiple_llm_instances
    PRIVATE 
        _GLIBCXX_USE_CXX11_ABI=0 
        USE_CXX11_ABI=OFF
)
else()
target_compile_definitions(multiple_llm_instances
    PRIVATE 
        _GLIBCXX_USE_CXX11_ABI=1 
        USE_CXX11_ABI=ON
)
endif()


set_target_properties(multiple_llm_instances PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "/usr/local/xllm/lib"
    RPATH "/usr/local/xllm/lib"
)