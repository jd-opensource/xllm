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

target_link_libraries(mooncake_store PUBLIC transfer_engine cachelib_memory_allocator)
