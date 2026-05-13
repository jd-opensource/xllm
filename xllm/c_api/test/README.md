# c_api/test — `xllm_test` 说明

## 作用

`xllm_test` 是一个基于 **brpc** 的小型服务进程，用于在 **RPC 层** 验证 xLLM 的 **C API**（`xllm/c_api/llm.h` 或 `rec.h`）：

- 对客户端暴露 **一个** RPC：`Inference(XLLM_Request) -> XLLM_Response`（定义见 `xllm_test.proto`）。
- 根据请求里的 **`call_function`** 字符串，转发到对应的 C API（例如 `xllm_llm_completions`、`xllm_rec_text_completions` 等）。
- 请求/响应中的结构与 `types.h` 对齐，由 `utils.cpp` 在 **Protobuf** 与 **C 结构体** 之间做转换。

**注意**：一次进程只加载 **一种** 后端，由 **`--backend`** 决定：

| `--backend` | 使用的 C API | 仅有效的 `call_function` 前缀 |
|-------------|--------------|--------------------------------|
| `llm`       | `llm.h`      | `xllm_llm_*`                   |
| `rec`       | `rec.h`      | `xllm_rec_*`                   |

若后端与 `call_function` 不匹配（例如在 `rec` 模式下调用 `xllm_llm_completions`），会返回错误（例如 handler 为空）。

---

## 依赖与前置条件

1. **已安装的 C API 头文件与 `libxllm.so`**  
   默认按 **`/usr/local/xllm/include`** 与 **`/usr/local/xllm/lib`** 查找（与 `CMakeLists.txt` 一致）。  
   若尚未安装，可在仓库内执行 `xllm/c_api/install.sh`（或你们环境约定的安装方式）。

2. **主工程已构建出的 brpc**  
   `libbrpc.a`（及头文件）通常位于仓库根目录下类似路径：  
   `build/third_party/brpc/output/` 或  
   `build/<toolchain>/third_party/brpc/output/`（例如 `cmake.linux-aarch64-cpython-311`）。  
   CMake 会自动在 `build/*/third_party/brpc/output` 下搜索；若仍找不到，可设置：  
   `-DBRPC_ROOT=/path/to/.../third_party/brpc/output` 或环境变量 **`BRPC_ROOT`**。

3. **Protobuf、gflags、glog、leveldb、OpenSSL、Zlib**  
   推荐与主工程一致，使用 **vcpkg**（仓库根目录 `vcpkg.json`），配置时传入  
   `-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake`、  
   `-DVCPKG_MANIFEST_DIR=<xllm 仓库根目录>`。

---

## 编译

在 **`xllm/c_api/test`** 目录下新建构建目录并配置、编译（请将占位路径换成你本机路径）：

```bash
cd xllm/c_api/test

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_MANIFEST_DIR=/path/to/xllm

cmake --build build -j$(nproc)
```

生成可执行文件：**`build/xllm_test`**（具体路径以 CMake 生成位置为准）。

若 vcpkg 依赖已安装在主工程构建目录中，也可通过 **`CMAKE_PREFIX_PATH`** 指向  
`<主工程 build>/vcpkg_installed/<triplet>`，避免重复安装。

---

## 运行

1. 编辑示例 flags：**`xllm_test.flags`**（至少设置 **`--model_path`**、**`--devices`**，并按需设置 **`--backend=llm`** 或 **`--backend=rec`**）。

2. 启动服务：

```bash
/path/to/build/xllm_test --flagfile=/path/to/xllm/c_api/test/xllm_test.flags
```

或在命令行直接传参，例如：

```bash
./build/xllm_test \
  --backend=rec \
  --model_path=/path/to/model \
  --devices=auto \
  --port=8000
```

3. **监听地址**  
   - 默认使用 **`--port`**（如 `8000`）在 `0.0.0.0` 上监听。  
   - 若设置 **`--listen_addr=host:port`**，则优先使用该地址（与 `xllm_test.flags` 中注释一致）。

4. **调用方式**  
   任意支持 **brpc + 同一套 `xllm_test.proto`** 的客户端，向上述地址发起 **`XllmRecCapiService/Inference`**，在 **`XLLM_Request.call_function`** 中填入与当前 **`--backend`** 一致的 API 名称即可。

---

## 目录内主要文件

| 文件 | 说明 |
|------|------|
| `xllm_test.cpp` | brpc 服务入口、`Inference` 分发逻辑 |
| `xllm_test.proto` | RPC 与消息定义 |
| `utils.cpp` / `utils.h` | Protobuf ↔ C API 类型转换、gflags 定义 |
| `xllm_test.flags` | 示例运行参数 |
| `CMakeLists.txt` | 构建配置 |

---

## 常见问题

- **`brpc` / `libbrpc.a` 找不到**：先在仓库根目录完整配置并编译主工程，使 `third_party/brpc` 产物出现；或使用 `-DBRPC_ROOT`。  
- **链接或运行找不到 `libxllm.so`**：确认已安装到 **`/usr/local/xllm/lib`**，或自行修改 `CMakeLists.txt` 中的 include/lib 路径并设置 **`LD_LIBRARY_PATH`**。  
- **与主进程 `127.0.0.1:18899` 相关日志**：那是 xLLM **分布式 engine/worker** 的地址，与 `xllm_test` 的 **`--port` / `--listen_addr`** 无关；需按主工程文档单独启动 engine。
