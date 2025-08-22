# MTP Speculative Inference

## Background
MTP (Multi-Token Prediction) is an innovative inference acceleration technique that addresses efficiency bottlenecks in large language model generation. By incorporating specialized pre-training designs, MTP provides efficient draft token prediction capabilities during inference, significantly improving generation speed. Its core value lies in balancing inference efficiency with output quality, offering an optimal solution for long-sequence generation problems in LLMs, ultimately optimizing inference performance.

## Key Features
MTP offers the following core acceleration capabilities:

- **Efficient Draft Generation**: Uses a lightweight MTP architecture to rapidly generate draft tokens that serve as input for the main model's verification, dramatically reducing computation overhead compared to traditional autoregressive generation.

- **Batch Verification Mechanism**: The main model can simultaneously verify multiple MTP-generated draft tokens in batch, rather than processing them sequentially, significantly boosting inference speed.

- **High Sampling Accuracy**: MTP solves the critical pain point of low token acceptance rates in post-training draft modules (like Eagle and Medusa). By optimizing draft generation during pre-training, MTP produces tokens with higher accuracy, reducing the verification burden on the main model.

- **Reduced Inference Latency**: By pre-generating multiple potential subsequent tokens, MTP effectively decreases cumulative latency during long-text generation, creating a smoother user experience.

- **Optimized Resource Consumption**: Compared to other inference acceleration techniques, MTP maintains acceleration effects while requiring fewer additional computational resources, making it suitable for deployment in resource-constrained environments.

MTP technology provides a novel efficiency optimization solution for LLM inference, particularly well-suited for real-time applications requiring rapid responses, representing an important direction in language model inference optimization.

!!! note "Model Support"
    Currently only supports Deepseek's MTP architecture. Support for other models will be added in future updates.

## Usage Example

### Export Model
```bash
./tools/export_deepseek_mtp.py --input-dir /path/to/DeepSeek-V3 --output-dir /path/to/DeepSeek-V3-mtp
```
Input model reference:[Deepseek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)


### Launch Script
```bash
MODEL_PATH="/models/DeepSeek-V3"
DRAFT_MODEL_PATH="/models/DeepSeek-V3-MTP"
MASTER_NODE_ADDR="127.0.0.1:42123"
START_PORT=13222
START_DEVICE=0
LOG_DIR="log"
NNODES=16

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup ./xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --draft_model $DRAFT_MODEL_PATH \
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens 1 \
    --max_memory_utilization=0.90 \
    --max_tokens_per_batch=10000 \
    --max_seqs_per_batch=256 \
    --enable_mla=true \
    --block_size=128 \
    --ep_size=1 \
    --dp_size=1 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --node_rank=$i > $LOG_FILE 2>&1 &
  sleep 0.5
done
```
# Performance Data
Based on ShareGPT dataset with input length=2500, output length=1500, total requests=80.

| method    | Concurrency | Mean TPOT(ms) | Mean TTFT(ms) | Output Tokens/s | Total Tokens/s |
|:---------:|:-----------:|:-------------:|:-------------:|:---------------:|:--------------:|
| baseline  |      1      |     40.61     |    141.80     |      24.20      |     65.77      |
| mtp       |      1      |     28.33     |    142.35     |      35.19      |     95.52      |
| baseline  |      2      |     42.69     |    178.59     |      45.16      |    122.74      |
| mtp       |      2      |     29.81     |    187.97     |      64.75      |    175.78      |
| baseline  |      4      |     46.18     |    172.34     |      79.83      |    216.96      |
| mtp       |      4      |     33.54     |    194.22     |     111.18      |    301.81      |
| baseline  |      8      |     53.16     |    181.49     |     110.68      |    300.81      |
| mtp       |      8      |     40.99     |    203.37     |     154.46      |    419.34      |
| baseline  |     16      |     68.50     |    213.89     |     143.81      |    390.84      |
| mtp       |     16      |     57.04     |    254.99     |     201.89      |    548.04      |
| baseline  |     20      |     74.72     |    228.80     |     154.77      |    420.65      |
| mtp       |     20      |     61.73     |    264.34     |     206.24      |    559.84      |
| baseline  |     40      |    119.68     |    559.32     |     180.22      |    489.80      |
| mtp       |     40      |    105.70     |    544.54     |     252.91      |    686.74      |
| baseline  |     80      |    180.89     |   2996.21     |     192.09      |    522.06      |
| mtp       |     80      |    152.19     |   2163.72     |     278.07      |    755.12      |