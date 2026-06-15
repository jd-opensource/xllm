# GeGraphExecutor 设计文档

## 目标
为 xLLM 框架新增 `GeGraphExecutorImpl`，使用华为 Ascend GE 接口执行 epair 文件加载的预编译 Graph 对象，实现最小化架构改动。

## 设计决策

### 关键约束
- 继承 `ExecutorImpl` 抽象基类（不继承 `BaseExecutorImpl`）
- 接受 `nullptr` 的 `CausalLM* model` 参数
- 单个 Graph 方案（不分桶），epair 支持动态 shape
- epair 文件路径：`model_path + "/model.epair"`
- Graph ID 固定为 `"main_graph"`
- 错误处理：宽松策略（LOG ERROR 不中断）
- 不修改现有 `WorkerImpl` 架构，通过 `options_.model_path()` 获取模型路径

### 用户确认
- **节点命名**：固定名称（`input_ids`, `position_ids`, `past_key_values`, `logits` 等）
- **Session 管理**：`GeGraphManager` 单例类中
- **编译时机**：构造函数中立即编译
- **KV Cache**：外部传入 KV Cache tensor（作为 Graph 输入/输出）

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
│                GeGraphManager (单例)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  session_: std::shared_ptr<ge::Session>              │   │
│  │  compiled_graphs_: std::unordered_map<...>           │   │
│  │  device_id_: uint64_t                                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               GE Runtime (Ascend)                            │
│  ┌───────────────────────┐  ┌───────────────────────────┐  │
│  │  ge::Session          │  │  ge::Graph                │  │
│  │  - AddGraph()         │  │  - Load from epair        │  │
│  │  - CompileGraph()     │  │  - Execute with stream    │  │
│  │  - ExecuteGraph...()  │  │                           │  │
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
- 管理 `ge::Session` 生命周期
- 缓存已编译的 Graph
- 分配 Graph ID
- 初始化/销毁 GE 和 ACL 运行时

**位置**：`xllm/core/runtime/ge_graph_manager.h`

```cpp
namespace xllm {
namespace core {

class GeGraphManager {
public:
    static GeGraphManager& Instance();
    
    // 初始化 GE/ACL 运行时
    ge::Status Initialize(uint64_t device_id);
    
    // 销毁运行时
    void Finalize();
    
    // 编译 Graph（如果未编译）
    ge::Status CompileGraph(const std::string& graph_key, 
                           const std::string& epair_path,
                           uint32_t& graph_id);
    
    // 执行 Graph
    ge::Status ExecuteGraph(uint32_t graph_id,
                           const std::vector<gert::Tensor>& inputs,
                           std::vector<gert::Tensor>& outputs);
    
    // 获取 Session
    std::shared_ptr<ge::Session> GetSession() const { return session_; }
    
    // 获取 Device ID
    uint64_t GetDeviceId() const { return device_id_; }
    
private:
    GeGraphManager() = default;
    ~GeGraphManager();
    
    GeGraphManager(const GeGraphManager&) = delete;
    GeGraphManager& operator=(const GeGraphManager&) = delete;
    
    // 成员变量
    std::shared_ptr<ge::Session> session_;
    std::unordered_map<std::string, uint32_t> compiled_graphs_;  // graph_key -> graph_id
    std::unordered_map<uint32_t, ge::Graph> graph_cache_;        // graph_id -> graph
    uint64_t device_id_ = 0;
    uint32_t next_graph_id_ = 0;
    bool initialized_ = false;
    std::mutex mutex_;
};

}  // namespace core
}  // namespace xllm
```

**实现细节**：

#### Initialize()
```cpp
ge::Status GeGraphManager::Initialize(uint64_t device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        return ge::SUCCESS;
    }
    
    device_id_ = device_id;
    
    // 1. 初始化 GE
    std::map<ge::AscendString, ge::AscendString> global_options = {
        {ge::AscendString("ge.graphRunMode"), ge::AscendString("0")},
        {ge::AscendString("ge.exec.deviceId"), ge::AscendString(std::to_string(device_id).c_str())}
    };
    if (ge::GEInitialize(global_options) != ge::SUCCESS) {
        LOG(ERROR) << "GEInitialize failed";
        return ge::FAILED;
    }
    
    // 2. 初始化 ACL
    if (aclInit(nullptr) != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclInit failed";
        return ge::FAILED;
    }
    
    // 3. 创建 Session
    std::map<ge::AscendString, ge::AscendString> session_options = {
        {ge::AscendString("ge.exec.precision_mode"), ge::AscendString("must_keep_origin_dtype")},
        {ge::AscendString("ge.session_device_id"), ge::AscendString(std::to_string(device_id).c_str())}
    };
    session_ = std::make_shared<ge::Session>(session_options);
    
    initialized_ = true;
    return ge::SUCCESS;
}
```

#### CompileGraph()
```cpp
ge::Status GeGraphManager::CompileGraph(const std::string& graph_key,
                                        const std::string& epair_path,
                                        uint32_t& graph_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查是否已编译
    auto it = compiled_graphs_.find(graph_key);
    if (it != compiled_graphs_.end()) {
        graph_id = it->second;
        return ge::SUCCESS;
    }
    
    // 加载 epair
    ge::Graph graph;
    td::EpairModelLoader loader(epair_path.c_str());
    if (loader.GetGEGraph(&graph) != td::SUCCESS) {
        LOG(ERROR) << "Failed to load epair: " << epair_path;
        return ge::FAILED;
    }
    
    // 分配 Graph ID
    graph_id = next_graph_id_++;
    
    // AddGraph
    if (session_->AddGraph(graph_id, graph) != ge::SUCCESS) {
        LOG(ERROR) << "AddGraph failed";
        return ge::FAILED;
    }
    
    // CompileGraph
    if (session_->CompileGraph(graph_id) != ge::SUCCESS) {
        LOG(ERROR) << "CompileGraph failed";
        return ge::FAILED;
    }
    
    // 缓存
    compiled_graphs_[graph_key] = graph_id;
    graph_cache_[graph_id] = std::move(graph);
    
    return ge::SUCCESS;
}
```

#### ExecuteGraph()
```cpp
ge::Status GeGraphManager::ExecuteGraph(uint32_t graph_id,
                                        const std::vector<gert::Tensor>& inputs,
                                        std::vector<gert::Tensor>& outputs) {
    // 1. Set Device
    if (aclrtSetDevice(device_id_) != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtSetDevice failed";
        return ge::FAILED;
    }
    
    // 2. Create Stream
    aclrtStream stream;
    if (aclrtCreateStream(&stream) != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtCreateStream failed";
        aclrtResetDevice(device_id_);
        return ge::FAILED;
    }
    
    // 3. LoadGraph
    if (session_->LoadGraph(graph_id, {}, stream) != ge::SUCCESS) {
        LOG(ERROR) << "LoadGraph failed";
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id_);
        return ge::FAILED;
    }
    
    // 4. H2D Copy
    std::vector<gert::Tensor> device_inputs;
    if (!ConstructInputDeviceTensor(inputs, device_inputs)) {
        LOG(ERROR) << "H2D copy failed";
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id_);
        return ge::FAILED;
    }
    
    // 5. ExecuteGraphWithStreamAsync
    std::vector<gert::Tensor> device_outputs;
    if (session_->ExecuteGraphWithStreamAsync(graph_id, stream, device_inputs, device_outputs) 
        != ge::SUCCESS) {
        LOG(ERROR) << "ExecuteGraphWithStreamAsync failed";
        FreeDevice(device_inputs);
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id_);
        return ge::FAILED;
    }
    
    // 6. Synchronize Stream
    if (aclrtSynchronizeStream(stream) != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtSynchronizeStream failed";
        FreeDevice(device_inputs);
        FreeDevice(device_outputs);
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id_);
        return ge::FAILED;
    }
    
    // 7. Construct Host Outputs (动态 shape)
    if (!ConstructHostOutputs(device_outputs, outputs)) {
        LOG(ERROR) << "ConstructHostOutputs failed";
        FreeDevice(device_inputs);
        FreeDevice(device_outputs);
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id_);
        return ge::FAILED;
    }
    
    // 8. D2H Copy
    if (!CopyD2H(device_outputs, outputs)) {
        LOG(ERROR) << "D2H copy failed";
        FreeDevice(device_inputs);
        FreeDevice(device_outputs);
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id_);
        return ge::FAILED;
    }
    
    // 9. Cleanup
    FreeDevice(device_inputs);
    FreeDevice(device_outputs);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id_);
    
    return ge::SUCCESS;
}
```

---

### 2. GeGraphExecutorImpl 类

**职责**：
- 加载 epair 文件并编译 Graph
- 转换 `torch::Tensor` ↔ `gert::Tensor`
- 执行 Graph 推理
- 处理输入/输出（包括 KV Cache）

**位置**：`xllm/core/runtime/ge_graph_executor_impl.h`

```cpp
namespace xllm {
namespace core {

class GeGraphExecutorImpl : public ExecutorImpl {
public:
    GeGraphExecutorImpl(CausalLM* model, 
                       ModelArgs args, 
                       Device device, 
                       Options options);
    
    ~GeGraphExecutorImpl() override;
    
    ModelOutput run(const std::vector<int>& tokens, 
                   int n_words,
                   const std::vector<std::pair<int, float>>* token_chooser,
                   const std::vector<int>* output_range) override;
    
    ModelOutput forward(const std::vector<int>& tokens, 
                        int n_words,
                        bool past_key_values_null,
                        const std::vector<int>* output_range) override;
    
private:
    // 转换 torch::Tensor → gert::Tensor (Host)
    bool ConvertTorchToGeTensor(const torch::Tensor& torch_tensor,
                               gert::Tensor& ge_tensor);
    
    // 转换 gert::Tensor (Host) → torch::Tensor
    bool ConvertGeToTorchTensor(const gert::Tensor& ge_tensor,
                               torch::Tensor& torch_tensor);
    
    // 准备 Graph 输入
    bool PrepareGraphInputs(const std::vector<int>& tokens,
                           std::vector<gert::Tensor>& inputs);
    
    // 处理 Graph 输出
    bool ProcessGraphOutputs(const std::vector<gert::Tensor>& outputs,
                            ModelOutput& result);
    
private:
    uint32_t graph_id_;
    std::string graph_key_;
    bool compiled_;
    
    // Graph 输入/输出节点名称（固定）
    static constexpr const char* INPUT_NODE_IDS = "input_ids";
    static constexpr const char* INPUT_NODE_POSITION_IDS = "position_ids";
    static constexpr const char* INPUT_NODE_PAST_KV = "past_key_values";
    static constexpr const char* OUTPUT_NODE_LOGITS = "logits";
    static constexpr const char* OUTPUT_NODE_PRESENT_KV = "present_key_values";
};

// 注册 Executor
REGISTER_EXECUTOR(ge, GeGraphExecutorImpl);

}  // namespace core
}  // namespace xllm
```

**实现细节**：

#### 构造函数
```cpp
GeGraphExecutorImpl::GeGraphExecutorImpl(CausalLM* model, 
                                         ModelArgs args, 
                                         Device device, 
                                         Options options)
    : ExecutorImpl(model, args, device, options), compiled_(false) {
    
    // 1. model 必须为 nullptr（因为继承自 ExecutorImpl，不调用 model_->forward()）
    if (model != nullptr) {
        LOG(ERROR) << "GeGraphExecutorImpl requires model to be nullptr";
        return;
    }
    
    // 2. 初始化 GeGraphManager
    uint64_t device_id = options.device_id();  // 假设 Options 有 device_id 字段
    if (GeGraphManager::Instance().Initialize(device_id) != ge::SUCCESS) {
        LOG(ERROR) << "Failed to initialize GeGraphManager";
        return;
    }
    
    // 3. 构造 epair 路径
    std::string epair_path = options.model_path() + "/model.epair";
    
    // 4. 生成 graph_key（基于 model_path）
    graph_key_ = options.model_path();  // 或使用其他唯一标识
    
    // 5. 编译 Graph
    if (GeGraphManager::Instance().CompileGraph(graph_key_, epair_path, graph_id_) 
        != ge::SUCCESS) {
        LOG(ERROR) << "Failed to compile graph: " << epair_path;
        return;
    }
    
    compiled_ = true;
    LOG(INFO) << "GeGraphExecutorImpl initialized, graph_id=" << graph_id_;
}
```

#### run() 实现
```cpp
ModelOutput GeGraphExecutorImpl::run(const std::vector<int>& tokens, 
                                     int n_words,
                                     const std::vector<std::pair<int, float>>* token_chooser,
                                     const std::vector<int>* output_range) {
    if (!compiled_) {
        LOG(ERROR) << "Graph not compiled";
        return ModelOutput();  // 返回空输出
    }
    
    // 1. 准备输入
    std::vector<gert::Tensor> inputs;
    if (!PrepareGraphInputs(tokens, inputs)) {
        LOG(ERROR) << "Failed to prepare inputs";
        return ModelOutput();
    }
    
    // 2. 执行 Graph
    std::vector<gert::Tensor> outputs;
    if (GeGraphManager::Instance().ExecuteGraph(graph_id_, inputs, outputs) 
        != ge::SUCCESS) {
        LOG(ERROR) << "Failed to execute graph";
        return ModelOutput();
    }
    
    // 3. 处理输出
    ModelOutput result;
    if (!ProcessGraphOutputs(outputs, result)) {
        LOG(ERROR) << "Failed to process outputs";
        return ModelOutput();
    }
    
    return result;
}
```

#### forward() 实现
```cpp
ModelOutput GeGraphExecutorImpl::forward(const std::vector<int>& tokens, 
                                         int n_words,
                                         bool past_key_values_null,
                                         const std::vector<int>* output_range) {
    // forward() 与 run() 的主要区别在于是否处理 KV Cache
    // 如果 past_key_values_null 为 false，则需要传入 KV Cache
    
    if (!compiled_) {
        LOG(ERROR) << "Graph not compiled";
        return ModelOutput();
    }
    
    // 准备输入（包括 KV Cache）
    std::vector<gert::Tensor> inputs;
    if (!PrepareGraphInputs(tokens, inputs)) {
        LOG(ERROR) << "Failed to prepare inputs";
        return ModelOutput();
    }
    
    // 如果需要传入 KV Cache
    if (!past_key_values_null) {
        // TODO: 从外部获取 KV Cache tensor
        // 这里需要与 RecWorkPipeline 的 KV Cache 管理机制集成
    }
    
    // 执行 Graph
    std::vector<gert::Tensor> outputs;
    if (GeGraphManager::Instance().ExecuteGraph(graph_id_, inputs, outputs) 
        != ge::SUCCESS) {
        LOG(ERROR) << "Failed to execute graph";
        return ModelOutput();
    }
    
    // 处理输出（包括 KV Cache）
    ModelOutput result;
    if (!ProcessGraphOutputs(outputs, result)) {
        LOG(ERROR) << "Failed to process outputs";
        return ModelOutput();
    }
    
    return result;
}
```

#### ConvertTorchToGeTensor()
```cpp
bool GeGraphExecutorImpl::ConvertTorchToGeTensor(const torch::Tensor& torch_tensor,
                                                  gert::Tensor& ge_tensor) {
    // 1. 转换 shape
    auto sizes = torch_tensor.sizes();
    gert::StorageShape shape;
    shape.SetDimNum(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
        shape.SetDim(i, sizes[i]);
    }
    
    // 2. 转换 dtype
    ge::DataType ge_dtype;
    switch (torch_tensor.dtype().toScalarType()) {
        case torch::kFloat32:
            ge_dtype = ge::DT_FLOAT;
            break;
        case torch::kFloat16:
            ge_dtype = ge::DT_FLOAT16;
            break;
        case torch::kInt32:
            ge_dtype = ge::DT_INT32;
            break;
        case torch::kInt64:
            ge_dtype = ge::DT_INT64;
            break;
        default:
            LOG(ERROR) << "Unsupported torch dtype: " << torch_tensor.dtype();
            return false;
    }
    
    // 3. 设置 Tensor 属性
    ge_tensor.GetShape() = shape;
    ge_tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    ge_tensor.SetDataType(ge_dtype);
    
    // 4. 分配 Host 内存并拷贝数据
    const size_t bytes = torch_tensor.numel() * torch_tensor.element_size();
    void* host_buf = nullptr;
    if (aclrtMallocHost(&host_buf, bytes) != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtMallocHost failed";
        return false;
    }
    
    // 5. 拷贝 torch tensor 数据到 host buffer
    std::memcpy(host_buf, torch_tensor.data_ptr(), bytes);
    
    // 6. 封装为 TensorData
    gert::TensorData td(host_buf, nullptr, bytes, gert::kOnHost);
    ge_tensor.SetData(std::move(td));
    
    return true;
}
```

#### ConvertGeToTorchTensor()
```cpp
bool GeGraphExecutorImpl::ConvertGeToTorchTensor(const gert::Tensor& ge_tensor,
                                                  torch::Tensor& torch_tensor) {
    // 1. 获取 shape
    auto shape = ge_tensor.GetStorageShape();
    std::vector<int64_t> sizes(shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        sizes[i] = shape.GetDim(i);
    }
    
    // 2. 转换 dtype
    torch::Dtype torch_dtype;
    switch (ge_tensor.GetDataType()) {
        case ge::DT_FLOAT:
            torch_dtype = torch::kFloat32;
            break;
        case ge::DT_FLOAT16:
            torch_dtype = torch::kFloat16;
            break;
        case ge::DT_INT32:
            torch_dtype = torch::kInt32;
            break;
        case ge::DT_INT64:
            torch_dtype = torch::kInt64;
            break;
        default:
            LOG(ERROR) << "Unsupported GE dtype: " << ge_tensor.GetDataType();
            return false;
    }
    
    // 3. 创建 torch tensor
    torch_tensor = torch::empty(sizes, torch::TensorOptions().dtype(torch_dtype));
    
    // 4. 拷贝数据
    const size_t bytes = ge_tensor.GetSize();
    std::memcpy(torch_tensor.data_ptr(), ge_tensor.GetAddr(), bytes);
    
    return true;
}
```

#### PrepareGraphInputs()
```cpp
bool GeGraphExecutorImpl::PrepareGraphInputs(const std::vector<int>& tokens,
                                             std::vector<gert::Tensor>& inputs) {
    // 1. 转换 input_ids
    torch::Tensor input_ids = torch::tensor(tokens, torch::kInt64).unsqueeze(0);  // [1, seq_len]
    gert::Tensor ge_input_ids;
    if (!ConvertTorchToGeTensor(input_ids, ge_input_ids)) {
        return false;
    }
    inputs.push_back(ge_input_ids);
    
    // 2. 转换 position_ids
    torch::Tensor position_ids = torch::arange(tokens.size(), torch::kInt64).unsqueeze(0);  // [1, seq_len]
    gert::Tensor ge_position_ids;
    if (!ConvertTorchToGeTensor(position_ids, ge_position_ids)) {
        return false;
    }
    inputs.push_back(ge_position_ids);
    
    // 3. 如果有 KV Cache，准备 past_key_values
    // TODO: 从外部获取 KV Cache
    
    return true;
}
```

#### ProcessGraphOutputs()
```cpp
bool GeGraphExecutorImpl::ProcessGraphOutputs(const std::vector<gert::Tensor>& outputs,
                                               ModelOutput& result) {
    if (outputs.empty()) {
        LOG(ERROR) << "No outputs from graph";
        return false;
    }
    
    // 1. 处理 logits（第一个输出）
    torch::Tensor logits;
    if (!ConvertGeToTorchTensor(outputs[0], logits)) {
        return false;
    }
    result.logits = logits;  // 假设 ModelOutput 有 logits 字段
    
    // 2. 如果有 KV Cache 输出（第二个输出）
    if (outputs.size() > 1) {
        torch::Tensor present_kv;
        if (!ConvertGeToTorchTensor(outputs[1], present_kv)) {
            return false;
        }
        // TODO: 保存 present_key_values 供下次推理使用
    }
    
    return true;
}
```

---

### 3. Tensor 转换工具类

**职责**：封装 `torch::Tensor` ↔ `gert::Tensor` 转换逻辑

**位置**：`xllm/core/runtime/ge_tensor_utils.h`

```cpp
namespace xllm {
namespace core {

class GeTensorUtils {
public:
    // torch::Tensor → gert::Tensor (Host)
    static bool TorchToGeHost(const torch::Tensor& torch_tensor, gert::Tensor& ge_tensor);
    
    // gert::Tensor (Host) → torch::Tensor
    static bool GeHostToTorch(const gert::Tensor& ge_tensor, torch::Tensor& torch_tensor);
    
    // gert::Tensor (Host) → gert::Tensor (Device)
    static bool CopyH2D(const gert::Tensor& host_tensor, gert::Tensor& device_tensor);
    
    // gert::Tensor (Device) → gert::Tensor (Host)
    static bool CopyD2H(const gert::Tensor& device_tensor, gert::Tensor& host_tensor);
    
private:
    // dtype 映射表
    static std::unordered_map<torch::Dtype, ge::DataType> torch_to_ge_dtype_;
    static std::unordered_map<ge::DataType, torch::Dtype> ge_to_torch_dtype_;
};

}  // namespace core
}  // namespace xllm
```

---

## 与 RecWorkPipeline 集成

### 集成方式

GeGraphExecutorImpl 通过工厂模式自动注册，RecWorkPipeline 无需修改：

```cpp
// xllm/core/runtime/executor_impl_factory.h
REGISTER_EXECUTOR(ge, GeGraphExecutorImpl);
```

当用户指定 `backend="ge"` 时，工厂会自动创建 `GeGraphExecutorImpl` 实例。

### 创建流程

```
xllm.cpp:115 (Options 构造)
    ↓
WorkerImpl::Init() (根据 backend 字符串创建 Executor)
    ↓
ExecutorImplFactory::Create(backend="ge")
    ↓
GeGraphExecutorImpl::GeGraphExecutorImpl(model=nullptr, args, device, options)
    ↓
GeGraphManager::Instance().Initialize()
    ↓
GeGraphManager::Instance().CompileGraph(model_path + "/model.epair")
```

---

## 文件结构

```
xllm/core/runtime/
├── ge_graph_manager.h              # GeGraphManager 单例类定义
├── ge_graph_manager.cpp            # GeGraphManager 实现
├── ge_graph_executor_impl.h       # GeGraphExecutorImpl 定义
├── ge_graph_executor_impl.cpp     # GeGraphExecutorImpl 实现
├── ge_tensor_utils.h              # Tensor 转换工具类定义
├── ge_tensor_utils.cpp            # Tensor 转换工具类实现
└── executor_impl_factory.h        # 添加 REGISTER_EXECUTOR(ge, GeGraphExecutorImpl)
```

---

## 依赖项

### 外部库
- `ge` (Graph Engine): 华为 Ascend GE 接口
- `acl` (Ascend Computing Language): Ascend 运行时库
- `gert::Tensor`: GE Runtime Tensor 接口
- `td::EpairModelLoader`: epair 文件加载器（来自 `torch_delegate` 库）

### 头文件
```cpp
#include <ge/ge_api.h>              // GE 接口
#include <acl.h>                    // ACL 接口
#include <acl_rt.h>                 // ACL Runtime 接口
#include "torch_delegate/epair_model_loader.h"  // epair 加载器
```

### CMake 链接
```cmake
target_link_libraries(xllm
    ge                          # GE 库
    acl                         # ACL 库
    torch_delegate              # epair 加载器库
)
```

---

## 错误处理策略

### 宽松错误处理
- 所有可能失败的操作都返回 `ge::FAILED` 或 `false`
- 使用 `LOG(ERROR)` 记录错误，不中断程序
- 返回空的 `ModelOutput` 对象

### 错误场景
1. **epair 文件不存在或损坏**
   - `EpairModelLoader::GetGEGraph()` 失败
   - 返回空 `ModelOutput`

2. **Graph 编译失败**
   - `AddGraph()` 或 `CompileGraph()` 失败
   - 返回空 `ModelOutput`

3. **Tensor 转换失败**
   - dtype 不支持
   - 内存分配失败
   - 返回空 `ModelOutput`

4. **Graph 执行失败**
   - `ExecuteGraphWithStreamAsync()` 失败
   - 流同步失败
   - 返回空 `ModelOutput`

---

## 性能优化点

### 1. Graph 缓存
- `GeGraphManager` 缓存已编译的 Graph，避免重复编译
- 多个 `GeGraphExecutorImpl` 实例共享同一个 Graph

### 2. Session 共享
- 全局共享 `ge::Session`，减少初始化开销

### 3. Stream 复用
- 可以考虑复用 `aclrtStream`，减少创建/销毁开销

### 4. 内存池
- 可以考虑实现 `gert::Tensor` 内存池，减少频繁的 `aclrtMalloc`/`aclrtFree`

---

## 测试计划

### 单元测试
1. `GeTensorUtils` 测试
   - `TorchToGeHost` / `GeHostToTorch` 转换正确性
   - dtype 映射正确性
   - shape 映射正确性

2. `GeGraphManager` 测试
   - 单例模式正确性
   - `Initialize` / `Finalize` 正确性
   - `CompileGraph` 缓存正确性
   - `ExecuteGraph` 正确性

3. `GeGraphExecutorImpl` 测试
   - 构造函数正确处理 `model == nullptr`
   - `run()` / `forward()` 返回正确结果
   - 错误处理正确性

### 集成测试
1. 端到端推理测试
   - 加载 epair 文件
   - 执行推理
   - 验证输出正确性

2. KV Cache 测试
   - 多轮推理 KV Cache 传递正确性

3. 多线程测试
   - 多个 `GeGraphExecutorImpl` 实例并发执行

---

## 后续工作

### Phase 1：基础实现（当前）
- [ ] 实现 `GeGraphManager` 单例类
- [ ] 实现 `GeTensorUtils` 工具类
- [ ] 实现 `GeGraphExecutorImpl` 基础版本（不支持 KV Cache）
- [ ] 单元测试
- [ ] 集成测试

### Phase 2：KV Cache 支持
- [ ] 与 RecWorkPipeline KV Cache 管理机制集成
- [ ] 实现 `past_key_values` / `present_key_values` 传递
- [ ] 测试多轮推理

### Phase 3：性能优化
- [ ] Stream 复用
- [ ] 内存池
- [ ] 异步执行优化

### Phase 4：高级功能
- [ ] 支持动态 batch
- [ ] 支持多 Graph（不同 batch size）
- [ ] 支持分布式推理

---

## 参考资料

### GE 接口文档
- 华为 Ascend GE API 参考
- `ge::Session` 接口文档
- `gert::Tensor` 接口文档

### 参考代码
- `D:\gitcode\xllm_demo\xllm_ge_backend\ge_runtime\run_async\src\model_inference.cc` - GE 接口使用示例
- `D:\gitcode\xllm_demo\xllm_ge_backend\ge_runtime\run_async\src\tensor_utils.cc` - Tensor 构造示例

### xLLM 相关文件
- `xllm/core/runtime/executor_impl.h` - ExecutorImpl 基类
- `xllm/core/runtime/base_executor_impl.cpp` - BaseExecutorImpl 实现
- `xllm/core/runtime/executor_impl_factory.h` - Executor 工厂
- `docs/zh/design/executor_design.md` - Executor 设计文档