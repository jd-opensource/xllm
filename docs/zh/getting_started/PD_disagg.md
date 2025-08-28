# PD分离部署

`xllm`支持PD分离部署，这需要与我们的另一个开源库[xllm service](https://github.com/jd-opensource/xllm-service)配套使用。
## xLLM Service依赖
首先，我们下载安装`xllm service`，与安装编译`xllm`类似：
```bash
git clone https://github.com/jd-opensource/xllm-service
cd xllm_service
git submodule init
git submodule update
```
`xllm_service`编译运行依赖[vcpkg](https://github.com/microsoft/vcpkg)和[etcd](https://github.com/etcd-io/etcd)。先确保在前面[编译xllm](./compile.md)时已经进行了`vcpkg`的安装且设置了`vcpkg`的路径：
```bash
export VCPKG_ROOT=/your/path/to/vcpkg
```
### etcd安装
使用etcd官方提供的[安装脚本](https://github.com/etcd-io/etcd/releases)进行安装，其脚本提供的默认安装路径是`/tmp/etcd-download-test/etcd`，我们可以手动修改其脚本中的安装路径，也可以运行完脚本之后手动迁移：
```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```
### xLLM Service编译
再执行编译:
```bash
mkdir -p build
cd build
cmake ..
make -j 8
cd ..
```
!!! warning "可能的错误"
    这里能会遇到关于`boost-locale`和`boost-interprocess`的安装错误：`vcpkg-src/packages/boost-locale_x64-linux/include: No such     file or directory`,`/vcpkg-src/packages/boost-interprocess_x64-linux/include: No such file or directory`
    我们使用`vcpkg`重新安装这些包:
    ```bash
    /path/to/vcpkg remove boost-locale boost-interprocess
    /path/to/vcpkg install boost-locale:x64-linux
    /path/to/vcpkg install boost-interprocess:x64-linux
    ```
## PD分离运行
启动etcd:
```bash 
./etcd-download-test/etcd --listen-peer-urls 'http://localhost:2390'  --listen-client-urls 'http://localhost:2389' --advertise-client-urls  'http://localhost:2391'
```
启动xllm service:
```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=0 \
ENABLE_XLLM_DEBUG_LOG=1 \
./build/xllm_service/xllm_master_serving \
    --etcd_addr="127.0.0.1:2389" \
    --http_server_port=9888 \
    --rpc_server_port=9889
```
启动Prefill节点：
```bash
bash start_pd.sh
```
启动Decode节点：
```bash
bash start_pd.sh decode
```
start_pd.sh脚本如下:
```bash title="start_pd.sh" linenums="1" hl_lines="34 56 30 53"
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU 版 PyTorch 路径
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch 安装路径
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # LibTorch 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/libtorch_npu/lib  # 添加 NPU LibTorch 库路径

# 清理日志和临时文件
\rm -rf core.*
\rm -rf ~/dynamic_profiling_socket_*

echo "$1"
if [ "$1" = "decode" ]; then
  # decode模式
  echo ">>>>> decode"
  export ASCEND_RT_VISIBLE_DEVICES=6,7  # 使用 NPU 设备 6 和 7

  ./xllm/build/xllm/core/server/xllm \
  --model /path/to/your/model \
  --max_memory_utilization 0.90 \
  --devices="npu:1" \
  --instance_role DECODE \
  --enable_disagg_pd=true \
  --enable_cuda_graph=false \
  --enable_prefix_cache=false \
  --backend=llm \
  --port=9996  \
  --xservice_addr=127.0.0.1:9889  \
  --host=127.0.0.1 \
  --disagg_pd_port=7780 \
  --cluster_id=1 \
  --device_ip=0.0.0.0 \
  --transfer_listen_port=26001 \
  --enable_service_routing=true
else
  # prefill模式
  echo ">>>>> prefill"
  export ASCEND_RT_VISIBLE_DEVICES=6,7  # 使用 NPU 设备 6 和 7 
  
  ./xllm/build/xllm/core/server/xllm \
  --model /path/to/your/model \
  --max_tokens_per_batch 102400  \
  --max_memory_utilization 0.90  \
  --devices="npu:0"  \
  --instance_role PREFILL \
  --enable_disagg_pd=true \
  --enable_cuda_graph=false \
  --enable_prefix_cache=false \
  --backend=llm \
  --port=9997  \
  --xservice_addr=127.0.0.1:9889  \
  --host=127.0.0.1 \
  --cluster_id=0 \
  --device_ip=0.0.0.0 \
  --transfer_listen_port=26000 \
  --disagg_pd_port=7781 \
  --enable_service_routing=true
fi
```
需要注意：

- PD分离在指定NPU Device的时候，需要对应的`device_ip`，这个每张卡是不一样的，具体的可以在非容器环境下的物理机器上执行下面命令看到,其呈现的`address_{i}=`后面的值就是对应`NPU {i}`的`device_ip`。
```bash
sudo cat /etc/hccn.conf
```
- `xservice_addr`需与`xllm_service`的`rpc_server_port`相同

测试命令和上面类似，注意`curl http://localhost:{PORT}/v1/chat/completions ...`的`PORT`选择为prefill节点的`port`或者`xllm service`的`http_server_port`。