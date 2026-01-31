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
### etcd安装
`xllm_service`依赖[etcd](https://github.com/etcd-io/etcd)，使用etcd官方提供的[安装脚本](https://github.com/etcd-io/etcd/releases)进行安装，其脚本提供的默认安装路径是`/tmp/etcd-download-test/etcd`，我们可以手动修改其脚本中的安装路径，也可以运行完脚本之后手动迁移：
```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```
### xLLM Service编译
先应用patch:
```bash
sh prepare.sh
```

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
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving --etcd_addr="127.0.0.1:12389" --http_server_port 28888 --rpc_server_port 28889 --tokenizer_path=/path/to/tokenizer_config_dir/
```

以Qwen2-7B为例

- 启动Prefill实例
    ```bash
    /path/to/xllm --model=path/to/Qwen2-7B-Instruct \
           --port=8010 \
           --devices="npu:0" \
           --master_node_addr="127.0.0.1:18888" \
           --enable_prefix_cache=false \
           --enable_chunked_prefill=false \
           --enable_disagg_pd=true \
           --instance_role=PREFILL \
           --etcd_addr=127.0.0.1:12389 \
           --transfer_listen_port=26000 \
           --disagg_pd_port=7777 \
           --node_rank=0 \
           --nnodes=1
    ```
- 启动Decode实例
    ```bash
    /path/to/xllm --model=path/to/Qwen2-7B-Instruct \
           --port=8020 \
           --devices="npu:1" \
           --master_node_addr="127.0.0.1:18898" \
           --enable_prefix_cache=false \
           --enable_chunked_prefill=false \
           --enable_disagg_pd=true \
           --instance_role=DECODE \
           --etcd_addr=127.0.0.1:12389 \
           --transfer_listen_port=26100 \
           --disagg_pd_port=7787 \
           --node_rank=0 \
           --nnodes=1
    ```
需要注意：

- PD分离需要读取`/etc/hccn.conf`文件，确保将物理机上的该文件映射到了容器中
- `etcd_addr`需与`xllm_service`的`etcd_addr`相同

测试命令和上面类似，注意`curl http://localhost:{PORT}/v1/chat/completions ...`的`PORT`选择为启动xLLM service的`http_server_port`。