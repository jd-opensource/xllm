# VLM Service Startup

This document describes how to start a VLM model service based on the xLLM inference engine.

## Single Device
Start the service by executing the following command in the main directory of the `xllm` project:
```bash
ASCEND_RT_VISIBLE_DEVICES=0 ./build/xllm/core/server/xllm --model=/path/to/Qwen2.5-VL-7B-Instruct  --port=12345  --max_memory_utilization 0.90
```

## Multiple Devices
Start the service by executing the following command in the main directory of the `xllm` project:
```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/xllm/core/server/xllm --model=/path/to/Qwen2.5-VL-7B-Instruct  --port=12345  --max_memory_utilization 0.90
```