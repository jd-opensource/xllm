include(cc_library)

add_subdirectory(sentencepiece)
add_subdirectory(minja)
add_subdirectory(brpc)
add_subdirectory(smhasher/src)
add_subdirectory(cpprestsdk)
set(CPPREST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpprestsdk/Release/include)
set(CPPREST_LIB ${CMAKE_BINARY_DIR}/third_party/cpprestsdk/Release/Binaries/libcpprest.a)
add_subdirectory(etcd_cpp_apiv3)

if(USE_NPU)
  add_subdirectory(spdlog)
  add_subdirectory(hccl_transfer/hccl_transfer)
endif()
add_subdirectory(Mooncake)

target_include_directories(mooncake_store PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/Mooncake/mooncake-transfer-engine/include
)

if(USE_ILU)
  if(TARGET cpprest)
      set_target_properties(cpprest PROPERTIES
          CXX_STANDARD 20
          CXX_STANDARD_REQUIRED ON
          CXX_EXTENSIONS OFF
      )
  endif()
  if(TARGET transfer_engine)
    target_compile_options(transfer_engine PRIVATE -std=c++20)
    set_target_properties(transfer_engine PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
    message(STATUS "Set C++20 for transfer_engine target")
  endif()
  if(TARGET SMHasherSupport)
      set_target_properties(SMHasherSupport PROPERTIES
          CXX_STANDARD 11
          CXX_STANDARD_REQUIRED ON
          CXX_EXTENSIONS OFF
      )
      message(STATUS "SMHasherSupport target found and configured")
  else()
      message(WARNING "SMHasherSupport target not found after adding smhasher")
  endif()
endif()

target_link_libraries(mooncake_store PUBLIC transfer_engine cachelib_memory_allocator)
