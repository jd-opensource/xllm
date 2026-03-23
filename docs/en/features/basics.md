# Basics

- xLLM uses a one-device-per-process architecture. Across multiple devices, RPC is used for function calls, and data communication during model computation uses device collective communication libraries.

- HCCL/LCCL are high-performance collective communication frameworks that provide data-parallel and model-parallel collective communication for both single-node multi-device and multi-node multi-device scenarios.
