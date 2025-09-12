# PD disaggregation

`xllm` supports PD disaggregation deployment, which requires integration with our other open-source library [xllm service](https://github.com/jd-opensource/xllm-service).

## xLLM Service Dependencies

First, download and install `xllm service`, similar to installing and compiling `xllm`:
```bash
git clone https://github.com/jd-opensource/xllm-service
cd xllm_service
git submodule init
git submodule update
```


### etcd Installation

`xllm_service` compilation and operation depend on [etcd](https://github.com/etcd-io/etcd).Use the [installation script](https://github.com/etcd-io/etcd/releases) provided by etcd for installation. The default installation path provided by the script is `/tmp/etcd-download-test/etcd`. You can either manually modify the installation path in the script or manually migrate after running the script:
```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```

### xLLM Service Compilation
Apply patch:
```bash
sh prepare.sh
```
Then execute the compilation:
```bash
mkdir -p build
cd build
cmake ..
make -j 8
cd ..
```

!!! warning "Potential Errors"
    You may encounter installation errors related to `boost-locale` and `boost-interprocess`: `vcpkg-src/packages/boost-locale_x64-linux/include: No such file or directory`, `/vcpkg-src/packages/boost-interprocess_x64-linux/include: No such file or directory`
    Reinstall these packages using `vcpkg`:
    ```bash
    /path/to/vcpkg remove boost-locale boost-interprocess
    /path/to/vcpkg install boost-locale:x64-linux
    /path/to/vcpkg install boost-interprocess:x64-linux
    ```

## PD Disaggregation Execution

Start etcd:
```bash 
./etcd-download-test/etcd --listen-peer-urls 'http://localhost:2390'  --listen-client-urls 'http://localhost:2389' --advertise-client-urls 'http://localhost:2391'
```

Start xllm service:
```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving --etcd_addr="127.0.0.1:12389" --http_server_port 28888 --rpc_server_port 28889 --tokenizer_path=/path/to/tokenizer_config_dir/
```

Taking Qwen2-7B as an example:

- Start Prefill Instance
    ``` shell linenums="1" hl_lines="3 9 10"
    ./xllm --model=path/to/Qwen2-7B-Instruct \
           --port=8010 \
           --devices="npu:0" \
           --master_node_addr="127.0.0.1:18888" \
           --enable_prefix_cache=false \
           --enable_chunked_prefill=false \
           --enable_disagg_pd=true \
           --instance_role=PREFILL \
           --etcd_addr=127.0.0.1:12389 \
           --device_ip=xx.xx.xx.xx \ # Replace with actual Device IP 
           --transfer_listen_port=26000 \
           --disagg_pd_port=7777 \
           --node_rank=0 \
           --nnodes=1
    ```
- Start Decode Instance 
    ``` shell linenums="1" hl_lines="3 9 10"
    ./xllm --model=path/to/Qwen2-7B-Instruct \
           --port=8020 \
           --devices="npu:1" \
           --master_node_addr="127.0.0.1:18898" \
           --enable_prefix_cache=false \
           --enable_chunked_prefill=false \
           --enable_disagg_pd=true \
           --instance_role=DECODE \
           --etcd_addr=127.0.0.1:12389 \
           --device_ip=xx.xx.xx.xx \ # Replace with actual Device IP 
           --transfer_listen_port=26100 \
           --disagg_pd_port=7787 \
           --node_rank=0 \
           --nnodes=1
    ```

Important notes:

- For PD disaggregation when specifying NPU Device, the corresponding `device_ip` is required. This is different for each device. You can see this by executing the following command on the physical machine outside the container environment. The value after `address_{i}=` displayed is the `device_ip` corresponding to `NPU {i}`.
```bash
sudo cat /etc/hccn.conf
```

- `etcd_addr` must match the `etcd_addr` of `xllm_service`

The test command is similar to above. Note that the `PORT` in `curl http://localhost:{PORT}/v1/chat/completions ...` should be the `port` of the `http_server_port` of xLLM service.