# VLM启动服务
本文档介绍基于xLLM推理引擎启动VLM模型服务

## 单卡
启动服务，在`xllm`工程主目录中执行下面命令：
```bash
ASCEND_RT_VISIBLE_DEVICES=0 ./build/xllm/core/server/xllm --model=/path/to/Qwen2.5-VL-7B-Instruct  --port=12345  --max_memory_utilization 0.90 --devices auto 
```

## 多卡
启动服务，在`xllm`工程主目录中执行下面命令：
```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/xllm/core/server/xllm --model=/path/to/Qwen2.5-VL-7B-Instruct  --port=12345  --max_memory_utilization 0.90 --devices auto
```