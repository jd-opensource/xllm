# 1. LLM Accuracy Test
## 1.1 Setup ais_bench
```bash
# Create a virtual environment for ais_bench using conda or uv
conda create --name ais_bench python=3.10 -y
conda activate ais_bench

# Clone ais_bench and install dependencies
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517

# Download the dataset and copy it to the ais_bench directory
cp -r /path/to/dataset  /path/to/benchmark/ais_bench/datasets
```

## 1.2 Modify Configuration
Modify the accuracy test configuration file according to your actual situation: `/path/to/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`. It is recommended to set the sampling parameters as follows:
```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/model/Qwen3-8B", # Model path
        model="Qwen3-8B", # Model name
        request_rate = 0,
        retry = 2,
        host_ip = "127.0.0.1",
        host_port = 19000, # xllm server port
        max_out_len = 32768, # Limit maximum model length
        batch_size=32,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.6,
            # top_k = -1,
            top_p = 0.95,
            # seed = None,
            # repetition_penalty = 1,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

## 1.3 Launch ais_bench
Before using ais_bench, you need to start the xllm server first. Use `ais_bench -h` to get parameter descriptions. The launch commands for gsm8k and ceval datasets are as follows:
```bash
# Using gsm8k dataset
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --dump-eval-details

# Using ceval dataset
ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt --merge-ds --dump-eval-details
```

We will integrate ais_bench and datasets (ceval and gsm8k) into the development image in the future. The ais_bench documentation and datasets are as follows:
* [ais_bench Documentation](https://ais-bench-benchmark.readthedocs.io/en/latest/index.html)
* [Datasets](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/all_params/datasets.html)

