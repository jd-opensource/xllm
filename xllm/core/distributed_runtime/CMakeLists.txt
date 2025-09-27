include(cc_library)

if(USE_NPU)
  include_directories(
    ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
  )
endif()

cc_library(
  NAME 
    distributed_runtime
  HDRS
    disagg_pd_service.h
    disagg_pd_service_impl.h
    dist_manager.h
    remote_worker.h
    worker_server.h
    worker_service.h
  SRCS
    disagg_pd_service.cpp
    disagg_pd_service_impl.cpp
    dist_manager.cpp
    remote_worker.cpp
    worker_server.cpp
    worker_service.cpp
  DEPS
    :api_service
    :runtime
    :util
    glog::glog
    proto::xllm_proto
    absl::flat_hash_set
    :parallel_state
)

cc_library(
  NAME
    worker_service
  HDRS
    worker_service.h
  SRCS
    worker_service.cpp
  DEPS
    :request
    absl::strings
    glog::glog
    proto::xllm_proto
)

cc_library(
  NAME
    collective_service
  HDRS
    collective_service.h
  SRCS
    collective_service.cpp
  DEPS
    absl::flat_hash_set
    glog::glog
    proto::xllm_proto
)

target_link_libraries(worker_service PRIVATE brpc leveldb::leveldb ZLIB::ZLIB protobuf::libprotobuf)

