# Basics

- xLLM uses a one-card-per-process architecture. Across multiple cards, RPC is used for function calls, and data communication during model computation uses device collective communication libraries.

- HCCL/LCCL are high-performance collective communication frameworks that provide data-parallel and model-parallel collective communication for both single-node multi-card and multi-node multi-card scenarios.
