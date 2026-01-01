# 1. LLM性能测试

首先需要启动xllm服务，参考[xllm启动脚本](getting_started/multi_node.md)，然后使用`benchmark`目录下的`benchmark_llm.py`脚本进行批量测试：
```bash
#!/bin/bash
set -e

python /path/to/benchmark_llm.py \
    --backend xllm \
    --dataset-name random \
    --random-range-ratio 1 \
    --num-prompt 128 \ # 请求总数
    --max-concurrency 2048 \ # 最大并发数
    --random-input 2048 \ # 输入长度
    --random-output 2048 \ # 输出长度
    --host 127.0.0.1 \
    --port 19000 \ # 启动xllm服务的端口号
    --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \ # 数据集路径
    --model /path/to/model/Qwen3-8B
```