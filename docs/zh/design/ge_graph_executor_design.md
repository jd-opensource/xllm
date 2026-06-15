# GeGraphExecutor 设计文档

## 目标
为 xLLM 框架新增 `GeGraphExecutorImpl`，使用华为 Ascend GE V2 接口（`GEInitializeV2` + `EpairModelLoader::CompileAndLoad` + `RunModelWithStreamAsync`）直接加载并执行 epair 模型，实现最小化架构改动。

## 设计决策

### 关键约束
- 继承 `ExecutorImpl` 抽象基类（不继承 `BaseExecutorImpl`）
- 接受 `nullptr` 的 `CausalLM* model` 参数
- 单个 Graph 方案（不分桶），epair 支持动态 shape
- epair 文件路径：`model_path + "/model.epair"`
- 错误处理：宽松策略（LOG ERROR 不中断，适合在线服务容错）
- 不修改现有 `WorkerImpl` 架构，通过 `options_.model_path()` 获取模型路径

### 用户确认
- **节点命名**：固定名称（由 epair 文件定义）
- **编译时机**：构造函数中立即编译（`CompileAndLoad`）
- **KV Cache**：外部传入 KV Cache tensor（作为 Graph 输入/输出）
- **Stream 管理**：复用 Worker 的 `compute_stream_`（通过 `c10_npu::getCurrentNPUStream()`）
- **内存管理**：手动管理 device memory（aclrtMalloc/aclrtFree）
- **输入 tensor 内存**：Device 内存，可直接用于 gert::Tensor，避免 H2D 拷贝
- **GE 初始化**：进程级别，GEInitializeV2 默认 deviceId=0
- **多卡场景**：每个 device 独立的 GE runtime，GeGraphExecutorImpl 与 deviceId 绑定
  - GEInitializeV2：默认 deviceId=0
  - EpairModelLoader：通过 session_device_id 指定真正的 deviceId
  - 每个 device 有独立的 Loader 集合
- **CompileAndLoad stream**：编译阶段 stream 参数可传 `nullptr`（静态 shape 场景的提前编排 kernel 与运行时流无关）
- **RunModelWithStreamAsync stream**：真正的 stream 在执行时传入（从 Worker 获取）

---

## Stream 管理

### xLLM Stream 管理机制

xLLM 通过 `StreamGuard` 设置当前 thread 的 stream，Executor 自动使用该 stream：

```cpp
// WorkerImpl (worker_impl.cpp:240-241)
prepare_stream_ = device_.get_stream_from_pool();  // 准备阶段
compute_stream_ = device_.current_stream();        // 计算阶段

// Worker 执行前设置 stream (llm_worker_impl.cpp:197)
c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

// Executor 获取当前 thread 的 stream
aclrtStream stream = c10_npu::getCurrentNPUStream(device_index).stream();
```

### GeGraphExecutorImpl Stream 策略

**核心原则**：
- **复用 Worker stream**：通过 `c10_npu::getCurrentNPUStream()` 获取
- **不创建/销毁**：避免每次 `aclrtCreateStream`/`aclrtDestroyStream`（节省 ~100us）
- **统一管理**：Worker 统一管理 stream 同步

**实现**：
```cpp
// 获取当前 thread 的 stream（由 Worker 设置）
aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

// 直接使用该 stream
loader.RunModelWithStreamAsync(stream, inputs, outputs);
aclrtSynchronizeStream(stream);
```

---

## 多卡场景设计

### 确认的多卡架构

根据 GE V2 接口特性，多卡场景采用以下设计：

1. **GEInitializeV2**：进程级别初始化，默认 deviceId=0
   ```cpp
   ge::GEInitializeV2({
       {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")}  // 默认值
   });
   ```

2. **EpairModelLoader**：每个 device 独立的 Loader，通过 `session_device_id` 指定真正的 deviceId
   ```cpp
   // CompileAndLoad 时指定 session_device_id
   loader.CompileAndLoad({
       {ge::AscendString("ge.session_device_id"), 
        ge::AscendString(std::to_string(device_id).c_str())}  // 真正的 deviceId
   }, stream);
   ```

3. **GeGraphExecutorImpl 与 deviceId 绑定**：每个 device 有独立的 Loader 集合
   - 每个 Worker 使用特定 device，创建对应 device 的 Executor
   - Executor 内部管理该 device 的 Loader

### GeGraphManager 多卡架构

**设计方案**：GeGraphManager 管理所有 device 的 Loader，按 device_id 分组缓存

```cpp
class GeGraphManager {
public:
    // 初始化 GE V2（进程级别，只执行一次）
    ge::Status Initialize();  // deviceId 默认为 0
    
    // 获取或创建指定 device 的 Loader（内部编译时 stream 传 nullptr）
    td::EpairModelLoader* GetOrCreateLoader(
        uint64_t device_id,              // 指定 device
        const std::string& graph_key,    // graph 标识
        const std::string& epair_path);  // 不需要 stream 入参
    
private:
    // 每个 device 有独立的 Loader 缓存
    // device_id -> (graph_key -> Loader)
    std::unordered_map<uint64_t, 
                       std::unordered_map<std::string, 
                                         std::unique_ptr<td::EpairModelLoader>>> 
        device_loaders_;
    
    bool ge_initialized_ = false;
    std::mutex mutex_;
};
```

### Initialize()（进程级别）

```cpp
ge::Status GeGraphManager::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // GEInitializeV2 只执行一次，deviceId 默认为 0
    if (ge_initialized_) {
        return ge::SUCCESS;
    }
    
    // 1. 初始化 GE V2（默认 deviceId=0）
    if (ge::GEInitializeV2({
            {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")}
        }) != ge::SUCCESS) {
        LOG(ERROR) << "GEInitializeV2 failed";
        return ge::FAILED;
    }
    ge_initialized_ = true;
    
    LOG(INFO) << "GeGraphManager initialized (process-level GE, deviceId=0 default)";
    return ge::SUCCESS;
}
```

### GetOrCreateLoader()（指定 device）

**stream 参数说明**：
- **编译阶段**：stream 参数传 `nullptr`（静态 shape 场景的提前编排 kernel 与运行时流无关）
- **执行阶段**：真正的 stream 在 `RunModelWithStreamAsync` 时传入

```cpp
td::EpairModelLoader* GeGraphManager::GetOrCreateLoader(
    uint64_t device_id,
    const std::string& graph_key,
    const std::string& epair_path) {  // 不需要 stream 参数
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 1. 检查该 device 是否有 Loader 缓存
    auto device_it = device_loaders_.find(device_id);
    if (device_it != device_loaders_.end()) {
        auto loader_it = device_it->second.find(graph_key);
        if (loader_it != device_it->second.end()) {
            return loader_it->second.get();  // 返回已缓存的 Loader
        }
    }
    
    // 2. 创建新 Loader
    auto loader = std::make_unique<td::EpairModelLoader>(epair_path.c_str());
    
    // 3. 编译并加载，指定 session_device_id（真正的 deviceId）
    // stream 参数传 nullptr（编译阶段不需要运行时流）
    if (loader->CompileAndLoad({
            {ge::AscendString("ge.session_device_id"), 
             ge::AscendString(std::to_string(device_id).c_str())}
        }, nullptr) != td::SUCCESS) {  // stream = nullptr
        LOG(ERROR) << "CompileAndLoad failed for epair: " << epair_path 
                   << ", device_id: " << device_id;
        return nullptr;
    }
    
    // 4. 缓存到对应 device 的 Loader 集合
    if (device_loaders_.find(device_id) == device_loaders_.end()) {
        device_loaders_[device_id] = {};  // 创建该 device 的 Loader 缓存
    }
    device_loaders_[device_id][graph_key] = std::move(loader);
    
    LOG(INFO) << "Created Loader for device " << device_id 
              << ", graph_key: " << graph_key;
    
    return device_loaders_[device_id][graph_key].get();
}
```

### GeGraphExecutorImpl 与 deviceId 绑定

```cpp
GeGraphExecutorImpl::GeGraphExecutorImpl(CausalLM* model,
                                         const ModelArgs& args,
                                         const torch::Device& device,
                                         const runtime::Options& options)
    : args_(args), device_(device), options_(options), compiled_(false) {
    
    // 1. device_id 来自 torch::Device
    device_id_ = device.index();  // 与 device 绑定
    
    // 2. 初始化 GeGraphManager（进程级别）
    if (GeGraphManager::Instance().Initialize() != ge::SUCCESS) {
        LOG(ERROR) << "Failed to initialize GeGraphManager";
        return;
    }
    
    // 3. 获取或创建该 device 的 Loader（编译阶段 stream=nullptr）
    std::string epair_path = options.model_path() + "/model.epair";
    graph_key_ = options.model_path();
    
    loader_ = GeGraphManager::Instance().GetOrCreateLoader(
        device_id_,     // 指定 device
        graph_key_, 
        epair_path);    // 不需要 stream 参数
    
    if (loader_ == nullptr) {
        LOG(ERROR) << "Failed to create loader for device " << device_id_;
        return;
    }
    
    compiled_ = true;
    LOG(INFO) << "GeGraphExecutorImpl initialized for device " << device_id_;
}
```

**关键点**：
- **编译阶段**：GetOrCreateLoader 内部调用 `CompileAndLoad({...}, nullptr)`
- **执行阶段**：run() 时通过 `getCurrentNPUStream` 获取真正的 stream，传入 `RunModelWithStreamAsync`

### 多卡场景流程图

```
进程启动
    ↓
GEInitializeV2 (deviceId=0)  ← 进程级别，只执行一次
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Worker 0 (device=0)                                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  GeGraphExecutorImpl(device_id=0)                      │  │
│  │    └─> GetOrCreateLoader(0, ...)                       │  │
│  │         └─> CompileAndLoad(session_device_id=0)        │  │
│  │              └─> Loader cache: device_loaders_[0]      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Worker 1 (device=1)                                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  GeGraphExecutorImpl(device_id=1)                      │  │
│  │    └─> GetOrCreateLoader(1, ...)                       │  │
│  │         └─> CompileAndLoad(session_device_id=1)        │  │
│  │              └─> Loader cache: device_loaders_[1]      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**关键点**：
- **GEInitializeV2**：进程级别，deviceId=0（默认值）
- **每个 Worker/Executor**：绑定特定 device_id
- **Loader 缓存**：按 device_id 分组，每个 device 有独立 Loader 集合
- **CompileAndLoad**：通过 `session_device_id` 指定真正的 deviceId，stream 参数传 `nullptr`
- **RunModelWithStreamAsync**：真正的 stream 在执行时传入（从 Worker 获取）

---

## Tensor 内存管理

### Executor 输入 Tensor 内存位置

**关键发现**：Executor 的输入 `torch::Tensor` 已经在 **Device 内存**上。

**证据**：
```cpp
// worker_impl.cpp:144-145（Worker 在调用 Executor 前）
move_tensor_to_device_if_needed(input.token_ids, device);
move_tensor_to_device_if_needed(input.positions, device);

// forward_params.h:418-465（ForwardInput::to(device)）
inputs.token_ids = safe_to(source_token_ids, device, true);
inputs.positions = safe_to(source_positions, device, true);
```

**结论**：
- `torch::Tensor.data_ptr()` 返回 **device memory pointer**（不是 host pointer）
- **可以直接用于 `gert::Tensor`，避免 H2D 拷贝**

### GE V2 Tensor 管理方式

参考 `main.cpp:153-195`，GE V2 要求用户手动管理 device memory：

```cpp
// 1. 创建 Device Tensor（输入）
void *MakeDeviceTensor(const InputBuffer &buf, gert::Tensor &t) {
    void *dev = nullptr;
    aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);  // 分配
    aclrtMemcpy(dev, bytes, buf.data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);  // H2D
    
    // 设置 tensor 属性
    for (auto d : buf.shape) {
        t.GetShape().MutableOriginShape().AppendDim(d);
        t.GetShape().MutableStorageShape().AppendDim(d);
    }
    t.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    t.SetDataType(dtype);
    t.SetData(gert::TensorData(dev, nullptr, bytes, gert::kOnDeviceHbm));
    return dev;
}

// 2. 创建 Device Tensor（输出，预分配）
void *MakeDeviceOutputTensor(const std::vector<int64_t> &dims,
                             ge::DataType dtype, gert::Tensor &t) {
    aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);  // 预分配
    t.SetData(gert::TensorData(dev, nullptr, bytes, gert::kOnDeviceHbm));
    return dev;
}

// 3. Detach Tensor（解除 memory 绑定）
void DetachTensor(gert::Tensor &t) {
    t.MutableTensorData().SetAddr(nullptr, nullptr);  // 解除绑定
}

// 4. 释放 Device Memory
void FreeTensors(std::vector<DevMem> &dev_mems) {
    for (auto &m : dev_mems) {
        aclrtFree(m.ptr);  // 手动释放
    }
}
```

**关键点**：
- **手动管理**：tensor 不持有 memory，需要外部管理
- **Detach before Free**：必须先 `DetachTensor` 再 `aclrtFree`
- **预分配输出**：用户必须预分配输出 tensor memory
- **Device memory 类型**：`gert::kOnDeviceHbm`

### 避免 H2D 拷贝的优化方案

**原方案（H2D 拷贝，性能差）**：
```cpp
// 1. 分配 device memory
aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);

// 2. H2D 拷贝（torch tensor 已经在 device，还要拷贝）
aclrtMemcpy(dev, bytes, torch_tensor.data_ptr(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);

// 3. 设置 gert::Tensor
ge_tensor.SetData(gert::TensorData(dev, nullptr, bytes, gert::kOnDeviceHbm));
```

**优化方案（直接使用 device memory）**：
```cpp
// 1. 直接获取 torch tensor 的 device pointer
void* dev_ptr = torch_tensor.data_ptr();  // 已经在 device memory

// 2. 直接设置 gert::Tensor（无需拷贝）
size_t bytes = torch_tensor.numel() * torch_tensor.element_size();
ge_tensor.SetData(gert::TensorData(dev_ptr, nullptr, bytes, gert::kOnDeviceHbm));

// 注意：不要 aclrtFree(dev_ptr)，torch tensor 会自动管理
```

**实现要点**：
- `torch::Tensor.data_ptr()` 在 NPU tensor 上返回 device memory pointer
- `gert::Tensor` 只是引用这块内存，不持有所有权
- torch tensor 会自动管理内存，不需要手动 `aclrtFree`

**限制**：
- 仅适用于输入 tensor（torch tensor 持有内存）
- 输出 tensor 仍需预分配（GE 执行后写入）

### GeGraphExecutorImpl Tensor 管理策略

**Phase 1 方案**：每次推理创建/销毁（简单，适合动态 shape）

**流程**：
```
1. Host → Device (aclrtMalloc + aclrtMemcpy H2D)
2. 预分配输出 tensors
3. RunModelWithStreamAsync
4. aclrtSynchronizeStream
5. Device → Host (aclrtMemcpy D2H)
6. DetachTensor + aclrtFree
```

---

## 架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      RecPipelineRuntime                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  executor_: std::unique_ptr<Executor>               │   │
│  │    └── GeGraphExecutorImpl (注册为 "ge" backend)    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               GeGraphManager (单例)                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  loaders_: map<graph_key, EpairModelLoader>          │   │
│  │  device_id_: uint64_t                                │   │
│  │  ge_initialized_: bool                               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               GE Runtime (Ascend GE V2)                     │
│  ┌───────────────────────┐  ┌───────────────────────────┐  │
│  │  EpairModelLoader     │  │  gert::Tensor             │  │
│  │  - CompileAndLoad()   │  │  - Device memory          │  │
│  │  - RunModel...Async() │  │  - Shape/Format/Dtype     │  │
│  └───────────────────────┘  └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 类层次结构

```
ExecutorImpl (抽象基类)
    ├── BaseExecutorImpl
    ├── VlmExecutorImpl
    ├── CudaGraphExecutorImpl
    ├── AclGraphExecutorImpl
    └── GeGraphExecutorImpl  <-- 新增
```

---

## 详细设计

### 1. GeGraphManager 类（单例）

**职责**：
- 管理 `EpairModelLoader` 缓存
- 初始化/销毁 GE V2 和 ACL 运行时

**位置**：`xllm/core/runtime/ge_graph_manager.h`

```cpp
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "torch_delegate/epair_model_loader.h"

namespace xllm {
namespace core {

class GeGraphManager {
public:
    static GeGraphManager& Instance();
    
    // 初始化 GE V2（进程级别，deviceId 默认为 0）
    ge::Status Initialize();
    
    // 销毁运行时
    void Finalize();
    
    // 获取或创建指定 device 的 Loader（多卡支持）
    // stream 参数：编译阶段传 nullptr，真正的 stream 在 RunModelWithStreamAsync 时传入
    td::EpairModelLoader* GetOrCreateLoader(
        uint64_t device_id,              // 指定 device
        const std::string& graph_key,
        const std::string& epair_path);  // 不需要 stream 参数
    
    bool IsInitialized() const { return ge_initialized_; }
    
private:
    GeGraphManager() = default;
    ~GeGraphManager();
    
    GeGraphManager(const GeGraphManager&) = delete;
    GeGraphManager& operator=(const GeGraphManager&) = delete;
    
    // 每个 device 有独立的 Loader 缓存
    std::unordered_map<uint64_t, 
                       std::unordered_map<std::string, 
                                         std::unique_ptr<td::EpairModelLoader>>> 
        device_loaders_;
    
    bool ge_initialized_ = false;
    std::mutex mutex_;
};

}  // namespace core
}  // namespace xllm
```

---

### 2. GeGraphExecutorImpl 类

**职责**：
- 加载 epair 文件并编译（通过 GeGraphManager）
- 转换 `torch::Tensor` ↔ `gert::Tensor`
- 执行 Graph 推理
- 管理 device memory

**位置**：`xllm/core/runtime/ge_graph_executor_impl.h`

```cpp
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "torch_delegate/epair_model_loader.h"
#include "runtime/ge_graph_manager.h"

namespace xllm {

struct DevMem {
    void *ptr;
    size_t bytes;
};

namespace core {

class GeGraphExecutorImpl : public ExecutorImpl {
public:
    GeGraphExecutorImpl(CausalLM* model,
                       const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);
    
    ~GeGraphExecutorImpl() override;
    
    ForwardInput prepare_inputs(Batch& batch) override;
    
    ModelOutput run(const torch::Tensor& tokens,
                   const torch::Tensor& positions,
                   std::vector<KVCache>& kv_caches,
                   const ModelInputParams& params) override;
    
private:
    // Tensor 转换
    bool TorchToDeviceTensor(const torch::Tensor& torch_tensor,
                            gert::Tensor& ge_tensor,
                            DevMem& mem);
    
    bool DeviceTensorToTorch(const gert::Tensor& ge_tensor,
                            torch::Tensor& torch_tensor);
    
    // 准备输入/输出
    bool PrepareDeviceInputs(const torch::Tensor& tokens,
                            const torch::Tensor& positions,
                            std::vector<gert::Tensor>& device_inputs,
                            std::vector<DevMem>& input_mems);
    
    bool PrepareDeviceOutputs(std::vector<gert::Tensor>& device_outputs,
                             std::vector<DevMem>& output_mems);
    
    bool CopyOutputsToHost(const std::vector<gert::Tensor>& device_outputs,
                          ModelOutput& result);
    
    void CleanupTensors(std::vector<gert::Tensor>& tensors,
                       std::vector<DevMem>& mems);
    
private:
    td::EpairModelLoader* loader_;
    std::string graph_key_;
    uint64_t device_id_;
    bool compiled_;
    
    ModelArgs args_;
    torch::Device device_;
    runtime::Options options_;
};

REGISTER_EXECUTOR("ge", GeGraphExecutorImpl);

}  // namespace core
}  // namespace xllm
```

#### 构造函数（多卡支持）
```cpp
GeGraphExecutorImpl::GeGraphExecutorImpl(CausalLM* model,
                                         const ModelArgs& args,
                                         const torch::Device& device,
                                         const runtime::Options& options)
    : args_(args), device_(device), options_(options), compiled_(false) {
    
    // 1. model 必须为 nullptr
    if (model != nullptr) {
        LOG(ERROR) << "GeGraphExecutorImpl requires model to be nullptr";
        return;
    }
    
    // 2. device_id 来自 torch::Device（与 device 绑定）
    device_id_ = device.index();
    
    // 3. 初始化 GeGraphManager（进程级别，deviceId=0）
    if (GeGraphManager::Instance().Initialize() != ge::SUCCESS) {
        LOG(ERROR) << "Failed to initialize GeGraphManager";
        return;
    }
    
    // 4. 获取或创建该 device 的 Loader（编译阶段 stream=nullptr）
    std::string epair_path = options.model_path() + "/model.epair";
    graph_key_ = options.model_path();
    
    loader_ = GeGraphManager::Instance().GetOrCreateLoader(
        device_id_,     // 指定 device（多卡支持）
        graph_key_, 
        epair_path);    // 不需要 stream 参数
    
    if (loader_ == nullptr) {
        LOG(ERROR) << "Failed to create loader for device " << device_id_;
        return;
    }
    
    compiled_ = true;
    LOG(INFO) << "GeGraphExecutorImpl initialized for device " << device_id_;
}
```

**关键点**：
- **device 绑定**：`device_id_` 来自 `torch::Device.index()`
- **进程级别初始化**：`Initialize()` 不传参数（默认 deviceId=0）
- **多卡 Loader**：`GetOrCreateLoader()` 传入 `device_id_`，指定该 device 的 Loader
- **编译阶段 stream**：GetOrCreateLoader 内部调用 `CompileAndLoad({...}, nullptr)`
- **执行阶段 stream**：run() 时通过 `getCurrentNPUStream` 获取真正的 stream

#### run()（优化版，避免 H2D 拷贝）
```cpp
ModelOutput GeGraphExecutorImpl::run(const torch::Tensor& tokens,
                                     const torch::Tensor& positions,
                                     std::vector<KVCache>& kv_caches,
                                     const ModelInputParams& params) {
    if (!compiled_) {
        LOG(ERROR) << "Graph not compiled";
        return ModelOutput();
    }
    
    // 1. 获取 stream
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_id_).stream();
    
    // 2. 准备输入 tensors（直接使用 device memory，无 H2D 拷贝）
    std::vector<gert::Tensor> device_inputs;
    
    // tokens tensor（已在 device memory）
    gert::Tensor ge_tokens;
    if (!TorchToDeviceTensor(tokens, ge_tokens)) {
        LOG(ERROR) << "Failed to convert tokens tensor";
        return ModelOutput();
    }
    device_inputs.push_back(ge_tokens);
    
    // positions tensor（已在 device memory）
    gert::Tensor ge_positions;
    if (!TorchToDeviceTensor(positions, ge_positions)) {
        LOG(ERROR) << "Failed to convert positions tensor";
        return ModelOutput();
    }
    device_inputs.push_back(ge_positions);
    
    // 注意：输入 tensor 不需要 DevMem，torch tensor 自动管理
    
    // 3. 准备输出 tensors（预分配）
    std::vector<gert::Tensor> device_outputs;
    std::vector<DevMem> output_mems;
    if (!PrepareDeviceOutputs(device_outputs, output_mems)) {
        LOG(ERROR) << "Failed to prepare device outputs";
        return ModelOutput();
    }
    
    // 4. 执行推理
    if (loader_->RunModelWithStreamAsync(stream, device_inputs, device_outputs) 
        != td::SUCCESS) {
        LOG(ERROR) << "RunModelWithStreamAsync failed";
        CleanupTensors(device_outputs, output_mems);  // 只清理输出
        return ModelOutput();
    }
    
    // 5. 同步
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        LOG(ERROR) << "aclrtSynchronizeStream failed";
        CleanupTensors(device_outputs, output_mems);
        return ModelOutput();
    }
    
    // 6. 获取输出
    ModelOutput result;
    if (!CopyOutputsToHost(device_outputs, result)) {
        LOG(ERROR) << "Failed to copy outputs to host";
        CleanupTensors(device_outputs, output_mems);
        return ModelOutput();
    }
    
    // 7. 清理输出 tensors（输入 tensors 不需要清理）
    CleanupTensors(device_outputs, output_mems);
    
    return result;
}
```

**关键优化**：
- **输入 tensor**：直接使用 torch tensor 的 device memory，无 H2D 拷贝
- **输出 tensor**：预分配 device memory，需要手动清理
- **性能提升**：减少 H2D 拷贝开销（~100us per tensor）

#### TorchToDeviceTensor()（优化版，避免 H2D 拷贝）
```cpp
bool GeGraphExecutorImpl::TorchToDeviceTensor(const torch::Tensor& torch_tensor,
                                              gert::Tensor& ge_tensor) {
    // 优化：torch_tensor 已经在 device memory，直接使用
    
    // 1. 检查 tensor 是否在 device 上
    if (torch_tensor.device().is_cpu()) {
        LOG(ERROR) << "Input tensor is on CPU, expected device";
        return false;
    }
    
    // 2. 获取 tensor 信息
    auto sizes = torch_tensor.sizes();
    size_t bytes = torch_tensor.numel() * torch_tensor.element_size();
    void* dev_ptr = torch_tensor.data_ptr();  // 直接获取 device pointer
    
    // 3. 设置 tensor 属性（无需拷贝）
    for (auto d : sizes) {
        ge_tensor.GetShape().MutableOriginShape().AppendDim(d);
        ge_tensor.GetShape().MutableStorageShape().AppendDim(d);
    }
    ge_tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    
    // 4. dtype 转换
    ge::DataType ge_dtype;
    switch (torch_tensor.scalar_type()) {
        case torch::kFloat32: ge_dtype = ge::DT_FLOAT; break;
        case torch::kFloat16: ge_dtype = ge::DT_FLOAT16; break;
        case torch::kInt32: ge_dtype = ge::DT_INT32; break;
        case torch::kInt64: ge_dtype = ge::DT_INT64; break;
        default:
            LOG(ERROR) << "Unsupported torch dtype: " << torch_tensor.scalar_type();
            return false;
    }
    ge_tensor.SetDataType(ge_dtype);
    
    // 5. 直接使用 torch tensor 的 device memory（关键优化）
    ge_tensor.SetData(gert::TensorData(dev_ptr, nullptr, bytes, gert::kOnDeviceHbm));
    
    // 注意：不要 aclrtFree(dev_ptr)，torch tensor 会自动管理内存
    
    return true;
}
```

**关键优化**：
- **无 H2D 拷贝**：直接使用 torch tensor 的 device memory
- **无内存分配**：不需要 `aclrtMalloc`
- **无内存释放**：torch tensor 自动管理，不需要 `aclrtFree`

**适用场景**：
- 输入 tensor（tokens, positions）
- Worker 已经把 tensor 移动到 device

**不适用场景**：
- 输出 tensor（仍需预分配）

#### PrepareDeviceOutputs()（输出 tensor 预分配）
```cpp
bool GeGraphExecutorImpl::PrepareDeviceOutputs(
    std::vector<gert::Tensor>& device_outputs,
    std::vector<DevMem>& output_mems) {
    
    // 输出 tensor 需要预分配（参考 main.cpp:376-394）
    
    // 1. 推断输出 shape（需要根据 epair 文件或固定规则）
    std::vector<int64_t> output_dims = {hidden_size};  // 示例
    
    // 2. 预分配输出 memory
    for (auto &dims : output_dims) {
        size_t n = 1;
        for (auto d : dims) n *= static_cast<size_t>(d);
        size_t bytes = n * sizeof(float);  // 假设 float
        
        void *dev = nullptr;
        if (aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            LOG(ERROR) << "aclrtMalloc for output failed";
            return false;
        }
        
        gert::Tensor t;
        for (auto d : dims) {
            t.GetShape().MutableOriginShape().AppendDim(d);
            t.GetShape().MutableStorageShape().AppendDim(d);
        }
        t.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
        t.SetDataType(ge::DT_FLOAT);
        t.SetData(gert::TensorData(dev, nullptr, bytes, gert::kOnDeviceHbm));
        
        device_outputs.push_back(t);
        output_mems.push_back({dev, bytes});
    }
    
    return true;
}
```

#### DeviceTensorToTorch()
```cpp
bool GeGraphExecutorImpl::DeviceTensorToTorch(const gert::Tensor& ge_tensor,
                                              torch::Tensor& torch_tensor) {
    // 1. 获取 shape
    auto shape = ge_tensor.GetStorageShape();
    std::vector<int64_t> sizes;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        sizes.push_back(shape.GetDim(i));
    }
    
    // 2. dtype 转换
    torch::ScalarType torch_dtype;
    switch (ge_tensor.GetDataType()) {
        case ge::DT_FLOAT: torch_dtype = torch::kFloat32; break;
        case ge::DT_FLOAT16: torch_dtype = torch::kFloat16; break;
        case ge::DT_INT32: torch_dtype = torch::kInt32; break;
        case ge::DT_INT64: torch_dtype = torch::kInt64; break;
        default:
            LOG(ERROR) << "Unsupported GE dtype: " << ge_tensor.GetDataType();
            return false;
    }
    
    // 3. 创建 torch tensor
    torch_tensor = torch::empty(sizes, 
        torch::TensorOptions().dtype(torch_dtype).device(torch::kCPU));
    
    // 4. D2H 拷贝
    size_t bytes = ge_tensor.GetSize();
    if (aclrtMemcpy(torch_tensor.data_ptr(), bytes,
                    ge_tensor.GetAddr(), bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        LOG(ERROR) << "aclrtMemcpy D2H failed";
        return false;
    }
    
    return true;
}
```

#### CleanupTensors()（只清理输出 tensor）
```cpp
void GeGraphExecutorImpl::CleanupTensors(std::vector<gert::Tensor>& tensors,
                                         std::vector<DevMem>& mems) {
    // 输入 tensor：torch tensor 自动管理，不需要清理
    
    // 输出 tensor：需要手动清理
    for (auto &t : tensors) {
        t.MutableTensorData().SetAddr(nullptr, nullptr);  // Detach
    }
    
    for (auto &m : mems) {
        if (m.ptr) {
            aclrtFree(m.ptr);  // 手动释放
            m.ptr = nullptr;
        }
    }
    
    tensors.clear();
    mems.clear();
}
```

**关键点**：
- **输入 tensor 不清理**：torch tensor 自动管理内存
- **输出 tensor 需清理**：手动预分配的 memory 需要 `DetachTensor` + `aclrtFree`

---

## 错误处理策略

### 为什么选择宽松策略

**对比严格策略**：
```cpp
// 严格策略：失败直接崩溃
CHECK(model != nullptr);

// 宽松策略：失败返回空结果
if (!compiled_) {
    LOG(ERROR) << "Graph not compiled";
    return ModelOutput();
}
```

**选择宽松策略的原因**：

1. **在线服务容错性**
   - GeGraphExecutor 主要用于在线推理服务
   - 单次推理失败不应导致整个服务崩溃
   - 用户请求可以快速失败并返回错误响应

2. **调试友好**
   - `LOG(ERROR)` 记录详细错误信息，便于排查问题
   - 不影响其他正常请求的处理

3. **与 AclGraphExecutorImpl 的差异**
   - AclGraphExecutorImpl 使用严格策略（`CHECK` 断言）
   - AclGraphExecutorImpl 主要用于离线训练/推理，崩溃可接受
   - GeGraphExecutor 用于在线服务，需要更高的可用性

4. **调用方责任**
   - 调用方需要检查返回结果是否为空
   - 空结果表示推理失败，需要降级处理或返回错误响应

**缺点**：
- 需要调用方检查返回结果
- 错误可能被忽略（如果调用方不检查）

---

## 文件结构

```
xllm/core/runtime/
├── ge_graph_manager.h              # GeGraphManager 单例类定义
├── ge_graph_manager.cpp            # GeGraphManager 实现
├── ge_graph_executor_impl.h       # GeGraphExecutorImpl 定义
├── ge_graph_executor_impl.cpp     # GeGraphExecutorImpl 实现
└── executor_impl_factory.h        # 添加 REGISTER_EXECUTOR("ge", GeGraphExecutorImpl)
```

---

## 依赖项

### 外部库
- `ge` (Graph Engine V2): 华为 Ascend GE V2 接口
- `acl` (Ascend Computing Language): Ascend 运行时库
- `gert::Tensor`: GE Runtime Tensor 接口
- `td::EpairModelLoader`: epair 文件加载器（来自 `torch_delegate` 库）
- `torch_npu`: PyTorch NPU 后端

### 头文件
```cpp
#include <ge/ge_api_v2.h>           // GE V2 接口
#include <acl/acl.h>                // ACL 接口
#include <torch_npu/csrc/core/npu/NPUStream.h>  // NPU Stream
#include "torch_delegate/epair_model_loader.h"  // epair 加载器
```

### CMake 链接
```cmake
target_link_libraries(xllm
    ge                          # GE V2 库
    acl                         # ACL 库
    torch_delegate              # epair 加载器库
    torch_npu                   # PyTorch NPU 后端
)
```

---

## 测试计划

### 单元测试
1. `GeGraphManager` 测试
   - 单例模式正确性
   - `Initialize` / `Finalize` 正确性
   - `GetOrCreateLoader` 缓存正确性

2. `GeGraphExecutorImpl` 测试
   - 构造函数正确处理 `model == nullptr`
   - `run()` 返回正确结果
   - Tensor 转换正确性
   - Device memory 管理正确性

### 集成测试
1. 端到端推理测试
   - 加载 epair 文件
   - 执行推理
   - 验证输出正确性

2. 多线程测试
   - 多个 `GeGraphExecutorImpl` 实例并发执行
   - Loader 缓存共享正确性

---

## 后续工作

### Phase 1：基础实现（当前）
- [ ] 实现 `GeGraphManager` 单例类
- [ ] 实现 `GeGraphExecutorImpl` 基础版本
- [ ] Tensor 内存管理
- [ ] 单元测试
- [ ] 集成测试

### Phase 2：KV Cache 支持
- [ ] 与 RecWorkPipeline KV Cache 管理集成
- [ ] KV Cache tensor 传递

### Phase 3：性能优化

**已实现的优化**：
- [x] **避免 H2D 拷贝**：输入 tensor 直接使用 torch tensor 的 device memory
  - Worker 已经把 tensor 移动到 device
  - 直接使用 `data_ptr()`，无需 `aclrtMemcpy H2D`
  - 性能提升：~100us per tensor

- [x] **Stream 复用**：使用 Worker 的 `compute_stream_`
  - 通过 `getCurrentNPUStream` 获取，无需创建/销毁
  - 性能提升：~100us per call

**待实现的优化**：
- [ ] **Device memory 缓存**：输出 tensor memory pool
  - 减少 `aclrtMalloc`/`aclrtFree` 开销
  - 需要考虑动态 shape

- [ ] **Tensor 缓存池**：缓存常用 tensor shape

- [ ] **异步执行优化**：异步 D2H 拷贝、流式处理

---

## 参考资料

### GE V2 接口文档
- 华为 Ascend GE V2 API 参考
- `ge::GEInitializeV2` / `ge::GEFinalizeV2`
- `td::EpairModelLoader` 接口

### 参考代码
- `D:\gitcode\xllm_demo\xllm_ge_backend\ge_runtime\main.cpp`
  - Line 284-292: GEInitializeV2
  - Line 323-331: EpairModelLoader::CompileAndLoad
  - Line 402-421: RunModelWithStreamAsync
  - Line 153-195: Device Tensor 创建
  - Line 213-215: DetachTensor
  - Line 203-210: FreeTensors

### xLLM 相关文件
- `xllm/core/runtime/executor_impl.h` - ExecutorImpl 基类
- `xllm/core/runtime/executor_impl_factory.h` - Executor 工厂
- `xllm/core/runtime/worker_impl.cpp:240-241` - Worker Stream 初始化
- `xllm/core/runtime/llm_worker_impl.cpp:197` - Worker StreamGuard
- `xllm/core/platform/stream.h` - Stream 抽象类
- `xllm/core/platform/device.cpp:252-264` - Device::current_stream()