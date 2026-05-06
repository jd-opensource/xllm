# xLLM MLU NUMA Binding Landing Notes

This file is the saved conversion output for
`.agents/skills/xllm-mlu-numa-binding`. The original skill files are left
unchanged.

## Reference Shape

PR jd-opensource/xllm#1014 added CUDA NUMA binding with this shape:

- Detect device-local NUMA nodes from a backend-visible device ordinal.
- Bind the engine process to a detected device-local NUMA node.
- Force workers on other NUMA nodes into spawned subprocesses.
- Bind spawned worker processes to their device-local NUMA node.
- Bind thread workers to their device-local NUMA node.

## Landing Scope

The MLU landing follows the same runtime shape and extends the shared NUMA path
instead of adding a separate MLU-only implementation.

Code paths:

- `xllm/core/platform/numa_utils.{h,cpp}`
  - Builds for CUDA and MLU.
  - Keeps CUDA detection through `cudaDeviceGetPCIBusId`.
  - Adds MLU detection through `cnrtDeviceGetPCIBusId`.
  - Reads `/sys/bus/pci/devices/<BDF>/numa_node`.
  - Logs explicit fallback when PCI query, sysfs read, or sysfs NUMA mapping
    fails.
- `xllm/core/platform/CMakeLists.txt`
  - Includes `numa_utils` and links `numa` for MLU.
- `xllm/core/distributed_runtime/dist_manager.cpp`
  - Enables engine process binding and multi-NUMA spawn isolation for MLU.
- `xllm/core/distributed_runtime/worker_server.cpp`
  - Enables thread worker NUMA binding for MLU.
- `xllm/core/distributed_runtime/spawn_worker_server/spawn_worker_server.cpp`
  - Enables spawned worker process NUMA binding for MLU.
- `xllm/core/distributed_runtime/CMakeLists.txt`
  - Builds the spawn worker helper for MLU.

## Device Index Finding

Current code passes `xllm::Device::index()` into NUMA detection. Under MLU,
`Device::set_device()` calls `torch_mlu::setDevice(index())`, and
`Device::device_count()` calls `torch_mlu::device_count()`. Spawned worker
startup reconstructs the device string as `mlu:<device_idx>` before calling
`set_device()`. The index used by the NUMA path is therefore the torch-MLU/CNRT
visible local ordinal, not a global rank.

## Verification Targets

- Mandatory build gate for xLLM code changes:
  `python setup.py build --device mlu` through `$xllm-build`.
- Source checks:
  - CUDA branch still calls `cudaDeviceGetPCIBusId`.
  - MLU branch calls `cnrtDeviceGetPCIBusId`.
  - No NUMA node is inferred from device id arithmetic.
  - No hard-coded CPU ranges are introduced.
  - `third_party/` changes are ignored.
