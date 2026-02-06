# 1. LLM精度测试
## 1.1 设置ais_bench
```bash
# 使用conda或uv为ais_bench创建虚拟环境
conda create --name ais_bench python=3.10 -y
conda activate ais_bench

# 下载ais_bench并安装依赖
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517

# 下载数据集并复制到ais_bench目录下
cp -r /path/to/dataset  /path/to/benchmark/ais_bench/datasets
```

## 1.2 修改配置
根据实际情况修改精度测试配置文件：`/path/to/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`，采样参数建议按如下代码设置：
```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/model/Qwen3-8B", # 模型路径
        model="Qwen3-8B", # 模型名称
        request_rate = 0,
        retry = 2,
        host_ip = "127.0.0.1",
        host_port = 19000, # xllm服务端端口
        max_out_len = 32768, # 限制模型最大长度
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

## 1.3 启动ais_bench
在使用ais_bench前需要先启动xllm服务。使用`ais_bench -h`能够获取参数含义，对于gsm8k和ceval数据集的启动命令如下：
```bash
# 使用gsm8k数据集
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --dump-eval-details

# 使用ceval数据集
ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt --merge-ds --dump-eval-details
```

我们会在未来将ais_bench和数据集（ceval和gsm8k）集成进开发镜像，ais_bench文档和数据集如下：
* [ais_bench文档](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/index.html)
* [数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/datasets.html)
