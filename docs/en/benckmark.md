# Performance Testing

## LLM
First, you need to start the xllm service. Refer to [xllm launch script](getting_started/launch_xllm.md), then use the `benchmark_llm.py` script in the `benchmark` directory for batch testing:

```bash
#!/bin/bash
set -e

python /path/to/benchmark_llm.py \
    --backend xllm \
    --dataset-name random \
    --random-range-ratio 1 \
    --num-prompt 128 \ # Total number of requests
    --max-concurrency 2048 \ # Maximum concurrency
    --random-input 2048 \ # Input length
    --random-output 2048 \ # Output length
    --host 127.0.0.1 \
    --port 18000 \ # Port number for xllm service
    --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \ # Dataset path
    --model /path/to/model/Qwen3-8B
```

