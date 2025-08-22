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


<p align="center">
    <img src="docs/assets/logo_with_llm.png" alt="xLLM" style="width:50%; height:auto;">
</p>

<p align="center">
| <a href="https://xllm.readthedocs.io/zh-cn/latest/"><b>Documentation</b></a> | 
</p>


## 1. 简介

**xLLM** 是一个高效且易用的开源智能推理框架，为模型在国产芯片上的推理提供企业级服务保障与高性能引擎计算能力。

#### 背景
当前，百亿至万亿参数规模的大语言模型正快速部署于智能客服、实时推荐、内容生成等核心业务场景，对国产计算硬件的高效支持已成为低成本推理部署的核心需求。现有推理引擎难以有效适配国产芯片等专用加速器的架构特性，硬件计算单元利用率低、MoE 架构下的负载不均衡与通信开销瓶颈、kv 缓存管理困难等问题，制约了请求的高效推理与系统的可扩展性。xLLM 推理引擎提升了 “通信 - 计算 - 存储” 全链路的资源利用效率，目前已支撑京东多场景、多模型的线上服务。

--- 

## 2. 核心特性
xLLM 提供了强大的智能计算能力，通过硬件系统的算力优化与算法驱动的决策控制，联合加速推理过程，实现高吞吐、低延迟的分布式推理服务。

**全图化/多层流水线执行编排**
- 框架调度层的异步解耦调度，减少计算空泡；
- 模型图层的计算和通信异步并行，重叠计算与通信；
- 算子内核层的异构计算单元深度流水，重叠计算与访存。

**动态shape的图执行优化**
- 基于参数化与多图缓存方法的动态尺寸适配，提升静态图灵活性；
- 受管控的显存池，保证地址安全可复用；
- 集成适配性能关键的自定义算子（如 *PageAttention*, *AllReduce*）。

**算子优化**
- *GroupMatmul* 优化，提升计算效率；
- *Chunked Prefill* 优化，支撑长序列输入。

**高效显存优化**
- 离散物理内存与连续虚拟内存的映射管理；
- 按需分配内存空间，减少内存碎片与浪费；
- 智能调度内存空间，增加内存页复用，减小分配延迟；
- 国产芯片相应算子适配。

**全局多级KV Cache管理**
- 多级缓存的kv智能卸载与预取；
- 以kv cache为中心的分布式存储架构；
- 多节点间kv的智能传输路由。

**算法优化**
- 投机推理优化，多核并行提升效率；
- MoE专家的动态负载均衡，实现专家分布的高效调整。


---

## 3. 代码结构
```
├── xllm/
|   : 主代码目录
│   ├── api_service/               # api服务化实现
│   ├── core/  
│   │   : xllm核心功能代码目录
│   │   ├── common/                
│   │   ├── distributed_runtime/   # 分布式PD服务实现
│   │   ├── framework/             # 引擎执行模块实现
│   │   ├── kernels/               # 国产芯片kernels适配实现
│   │   ├── layers/                # 模型层实现
│   │   ├── runtime/               # worker/executor角色实现
│   │   ├── scheduler/             # 批调度与PD调度实现
│   │   └── util/
│   ├── models/                    # 模型实现
│   ├── processors/                # 多模态模型预处理实现
│   ├── proto/                     # 通信协议
|   └── server/                    # xLLM服务实例
├── examples/                      # 服务调用示例
├── tools/                         # NPU Timeline生成工具
└── xllm.cpp                       # xLLM启动入口
```

当前支持模型列表：
- DeepSeek-V3/R1
- DeepSeek-R1-Distill-Qwen
- Kimi-k2
- Llama2/3
- MiniCPM-V
- Qwen2/2.5/QwQ
- Qwen2.5-VL
- Qwen3 / Qwen3-MoE

---


## 4. 快速开始
#### 安装
首先下载我们提供的镜像：
```bash
docker pull xllm/xllm-ai:xllm-0.6.0-dev-800I-A2-py3.11-openeuler24.03-lts
```
然后创建对应的容器
```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host  --device=/dev/davinci0  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /var/queue_schedule:/var/queue_schedule -v /mnt/cfs/9n-das-admin/llm_models:/mnt/cfs/9n-das-admin/llm_models -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /usr/local/sbin/:/usr/local/sbin/ -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/:/var/log/npu/slog -v /export/home:/export/home -w /export/home -v ~/.ssh:/root/.ssh  -v /var/log/npu/profiling/:/var/log/npu/profiling -v /var/log/npu/dump/:/var/log/npu/dump -v /home/:/home/  -v /runtime/:/runtime/  xllm/xllm-ai:xllm-0.6.0-dev-800I-A2-py3.11-openeuler24.03-lts
```

下载官方仓库与模块依赖：
```
git clone https://github.com/jd-opensource/xllm
cd xllm 
git submodule init
git submodule update
```
编译依赖vcpkg，我们编译的时候会默认下载vcpkg，也可以先提前下载vcpkg，然后设置环境变量:
```
git clone https://github.com/microsoft/vcpkg.git
export VCPKG_ROOT=/your/path/to/vcpkg
```
下载安装python依赖:
```
cd xllm
pip install -r cibuild/requirements-dev.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install --upgrade setuptools wheel
```

#### 编译
执行编译，在`build/`下生成可执行文件`build/xllm/core/server/xllm`：
```
python setup.py build
```
或直接用以下命令编译在`dist/`下生成whl包:
```
python setup.py bdist_wheel
```

#### 执行
运行例如如下命令启动xllm引擎：
```
./build/xllm/core/server/xllm \    # 启动 xllm 服务器程序
    --model=/path/to/your/llm  \   # 指定模型路径（需替换为实际路径）
    --backend=llm \                # 指定后端类型为 LLM
    --port=9977 \                  # 设置服务端口为 9977
    --max_memory_utilization 0.90  # 设置最大内存利用率为 90
```

---

## 5. 成为贡献者
您可以通过以下方法为 xLLM 作出贡献:

1. 在Issue中报告问题
2. 提供改进建议
3. 补充文档
    + Fork仓库
    + 修改文档
    + 提出pull request
4. 修改代码
    + Fork仓库
    + 创建新分支
    + 加入您的修改
    + 提出pull request

感谢您的贡献！ 🎉🎉🎉
如果您在开发中遇到问题，请参阅**[xLLM中文指南](https://xllm.readthedocs.io/zh-cn/latest)**

---

## 6. 社区支持
如果你在xLLM的开发或使用过程中遇到任何问题，欢迎在项目的Issue区域提交可复现的步骤或日志片段。
如果您有企业内部Slack，请直接联系xLLM Core团队。

欢迎沟通和联系我们:

<div align="center">
  <img src="xxx" alt="contact" width="50%" height="50%">
</div>

---

## 7. 致谢
本项目的实现得益于以下开源项目: 
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM) - 采用了ScaleLLM中构图方式和借鉴Runtime执行。
- [MindIE](https://www.hiascend.com/en/software/mindie) - 采用了MindIE中的算子优化、ATB。
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - 依赖构建了多级KV Cache管理机制。
- [brpc](https://github.com/apache/brpc) - 依赖brpc构建了高性能http service。
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) - 依赖tokenizers-cpp构建了c++ tokenizer。
- [safetensors](https://github.com/huggingface/safetensors) - 依赖其c binding safetensors能力。

感谢以下为xLLM作出贡献的[开发者](https://github.com/jd-opensource/xllm/graphs/contributors)
<a href="https://github.com/jd-opensource/xLLM/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/xllm" />
</a>

---

## 8. 许可证

[Apache License](LICENSE)

#### xLLM 由 JD.com 提供 
#### 感谢您对xLLM的关心与贡献!
