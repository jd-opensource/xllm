include(cc_binary)

set(SPAWN_WORKER_DEPS
  :models
  :model
  :distributed_runtime
  absl::strings
)

if(USE_NPU)
  list(APPEND SPAWN_WORKER_DEPS
    xllm_kernels
    ascendcl
    nnopbase
    atb
    atb_customize
    c_sec
    spdlog::spdlog
  )
endif()

cc_binary(
  NAME 
    spawn_worker
  HDRS
    spawn_worker_server.h
  SRCS
    spawn_worker_server.cpp
    spawn_worker_server_process.cpp
  DEPS
    :models
    :model
    :distributed_runtime
    absl::strings
    xllm_atb_layers
    ascendcl
    nnopbase
    atb
    atb_customize
    c_sec
    spdlog::spdlog
)

add_dependencies(export_module spawn_worker)
