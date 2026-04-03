include(cc_binary)

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
    $<$<BOOL:${USE_NPU}>:xllm_atb_layers>
    $<$<BOOL:${USE_NPU}>:ascendcl>
    $<$<BOOL:${USE_NPU}>:nnopbase>
    $<$<BOOL:${USE_NPU}>:atb>
    $<$<BOOL:${USE_NPU}>:atb_customize>
    $<$<BOOL:${USE_NPU}>:c_sec>
    spdlog::spdlog
)

add_dependencies(export_module spawn_worker)
