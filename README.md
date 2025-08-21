<!-- Copyright 2022 JD Co.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

[English](./README.md) | [中文](./README_zh.md)

## 1. Project Overview

**xLLM** is an efficient and user-friendly LLM intelligent inference framework that provides enterprise-level service guarantees and high-performance engine computing capabilities for model inference on domestic AI accelerators.


#### Background

LLM with parameter scales ranging from tens of billions to trillions are being rapidly deployed in core business scenarios such as intelligent customer service, real-time recommendation, and content generation. Efficient support for domestic computing hardware has become a core requirement for low-cost inference deployment. Existing inference engines struggle to effectively adapt to the architectural characteristics of dedicated accelerators like domestic chips. Performance issues such as low utilization of computing units, load imbalance and communication overhead bottlenecks under the MoE architecture, and difficulties in kv cache management have restricted the efficient inference of requests and the scalability of the system. The xLLM inference engine improves the resource efficiency of the entire  "communication-computation-storage" performance link and currently supports JD\.com's online services across multiple scenarios and with multiple models.

---

## 2. Core Features

**xLLM** delivers robust intelligent computing capabilities. By leveraging hardware system optimization and algorithm-driven decision control, it jointly accelerates the inference process, enabling high-throughput, low-latency distributed inference services.

**Full Graph Pipeline Execution Orchestration**
- Asynchronous decoupled scheduling at the requests scheduling layer, to reduce computational bubbles.
- Asynchronous parallelism of computation and communication at the model graph layer, overlapping computation and communication.
- Pipelining of heterogeneous computing units at the operator kernel layer, overlapping computation and memory access.

**Graph Optimization for Dynamic Shapes**
- Dynamic shape adaptation based on parameterization and multi-graph caching methods to enhance the flexibility of static graph.
- Controlled tensor memory pool to ensure address security and reusability.
- Integration and adaptation of performance-critical custom operators (e.g., *PageAttention*, *AllReduce*).

**Kernel Optimization**
- *GroupMatmul* optimization to improve computational efficiency.
- *Chunked Prefill* optimization to support long-sequence inputs.

**Efficient Memory Optimization**
- Mapping management between discrete physical memory and continuous virtual memory.
- On-demand memory allocation to reduce memory fragmentation.
- Intelligent scheduling of memory pages to increase memory reusability.
- Adaptation of corresponding operators for domestic accelerators.

**Global KV Cache Management**
- Intelligent offloading and prefetching of KV in hierarchical caches.
- KV cache-centric distributed storage architecture.
- Intelligent KV routing among computing nodes.

**Algorithm-driven Acceleration**
- Speculative decoding optimization to improve efficiency through multi-core parallelism.
- Dynamic load balancing of MoE experts to achieve efficient adjustment of expert distribution.

---

## 3. Code Architecture
```
├── xllm/
|   : main source folder
│   ├── api_service/               # code for api services
│   ├── core/  
│   │   : xllm core features folder
│   │   ├── common/                
│   │   ├── distributed_runtime/   # code for distributed and pd serving
│   │   ├── framework/             # code for execution orchestration
│   │   ├── kernels/               # adaption for npu kernels adaption
│   │   ├── layers/                # model layers impl
│   │   ├── runtime/               # code for worker and executor
│   │   ├── scheduler/             # code for batch and pd scheduler
│   │   └── util/
│   ├── models/                    # models impl
│   ├── processors/                # code for vlm pre-processing
│   ├── proto/                     # communication protocol
|   └── server/                    # xLLM server
├── examples/                      # examples of calling xLLM
├── tools/                         # code for npu time generations
└── xllm.cpp                       # entrypoint of xLLM
```

Supported models list:
- DeepSeek-V3/R1
- DeepSeek-R1-Distill-Qwen
- Kimi-k2
- Llama2/3
- MiniCPM-V
- Qwen2/2.5/QwQ
- Qwen2.5-VL
- Qwen3 / Qwen3-MoE


---

## 4. Quick Start
#### Installation
Install official repo and submodules：
```
git clone https://github.com/jd-opensource/xllm
cd xllm 
git submodule init
git submodule update
```
When compiling, `vcpkg` will be downloaded by default. Alternatively, you can download `vcpkg` in advance and then set the environment variable:
```
git clone https://github.com/microsoft/vcpkg.git
export VCPKG_ROOT=/your/path/to/vcpkg
```
Install python dependencies:
```
cd xllm
pip install -r cibuild/requirements-dev.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install --upgrade setuptools wheel
```

#### Compilation
When compiling, generate executable files `build/xllm/core/server/xllm` under `build/`:
```
python setup.py build
```
Or, compile directly using the following command to generate the whl package under `dist/`:
```
python setup.py bdist_wheel
```

#### Launch
Run the following command to start xLLM engine: 
```
./build/xllm/core/server/xllm \    # launch xllm server
    --model=/path/to/your/llm  \   # model path（to replace with your own path）
    --backend=llm \                # indicate the LLM backend
    --port=9977 \                  # set service port to 9977
    --max_memory_utilization 0.90  # set the maximal utilization of device memory
```

--- 

## 5. Contributing
There are several ways you can contribute to xLLM:

1. Reporting Issues (Bugs & Errors)
2. Suggesting Enhancements
3. Improving Documentation
    + Fork the repository
    + Add your view in document
    + Send your pull request
4. Writing Code
    + Fork the repository
    + Create a new branch
    + Add your feature or improvement
    + Send your pull request

We appreciate all kinds of contributions! 🎉🎉🎉
If you have problems about development, please check our document: **[Document](https://xllm.readthedocs.io/zh-cn/latest)**

---

## 6. Community & Support
If you encounter any issues along the way, you are welcomed to submit reproducible steps and log snippets in the project's Issues area, or contact the xLLM Core team directly via your internal Slack.

Welcome to contact us:

<div align="center">
  <img src="" alt="contact" width="50%" height="50%">
</div>

---

## 7. Acknowledgment

This project was made possible thanks to the following open-source projects:  
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM) - xLLM draws inspiration from ScaleLLM's graph construction method and references its runtime execution. 
- [MindIE](https://www.hiascend.com/en/software/mindie) - Develop xLLM kernels based on kernel optimization and ATB of MindIE.
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - Build xLLM hybrid KV cache management based on Mooncake.
- [brpc](https://github.com/apache/brpc) - Build high-performance http service based on brpc.
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) - Build C++ tokenizer based on tokenizers-cpp.
- [safetensors](https://github.com/huggingface/safetensors) - xLLM relies on the C binding safetensors capability.


Thanks to all the following [developers](https://github.com/jd-opensource/xllm/graphs/contributors) who have contributed to xLLM.
<a href="https://github.com/jd-opensource/xllm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/xllm" />
</a>

---

## 8. License
[Apache License](LICENSE)

#### xLLM is provided by JD.com 
#### Thanks for your Contributions!
