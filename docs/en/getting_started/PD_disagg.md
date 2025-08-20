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

`xllm_service` compilation and operation depend on [vcpkg](https://github.com/microsoft/vcpkg) and [etcd](https://github.com/etcd-io/etcd). First, ensure that `vcpkg` was installed during the previous [xllm compilation](./compile.md) and that the `vcpkg` path is set:
```bash
export VCPKG_ROOT=/your/path/to/vcpkg
```

### etcd Installation

Use the [installation script](https://github.com/etcd-io/etcd/releases) provided by etcd for installation. The default installation path provided by the script is `/tmp/etcd-download-test/etcd`. You can either manually modify the installation path in the script or manually migrate after running the script:
```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```

### xLLM Service Compilation

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
ENABLE_DECODE_RESPONSE_TO_SERVICE=0 \
ENABLE_XLLM_DEBUG_LOG=1 \
./build/xllm_service/xllm_master_serving \
    --etcd_addr="127.0.0.1:2389" \
    --http_server_port=9888 \
    --rpc_server_port=9889
```

Start Prefill node:
```bash
bash start_pd.sh
```

Start Decode node:
```bash
bash start_pd.sh decode
```

The start_pd.sh script is as follows:
```bash title="start_pd.sh" linenums="1" hl_lines="36 58"
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU version PyTorch path
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch installation path
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # LibTorch path
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # Ascend Toolkit environment variables
source /usr/local/Ascend/nnal/atb/set_env.sh  # ATB (Ascend Tensor Boost) environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/libtorch_npu/lib  # Add NPU LibTorch library path

# Clean up logs and temporary files
\rm -rf /root/atb/log/
\rm -rf /root/ascend/log/
\rm -rf core.*
\rm -rf ~/dynamic_profiling_socket_*

echo "$1"
if [ "$1" = "decode" ]; then
  # decode mode
  echo ">>>>> decode"
  export ASCEND_RT_VISIBLE_DEVICES=6,7  # Use NPU devices 6 and 7

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
  # prefill mode
  echo ">>>>> prefill"
  export ASCEND_RT_VISIBLE_DEVICES=6,7  # Use NPU devices 6 and 7 
  
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

Important notes:

- For PD disaggregation when specifying NPU Device, the corresponding `device_ip` is required. This is different for each card. You can see this by executing the following command on the physical machine outside the container environment. The value after `address_{i}=` displayed is the `device_ip` corresponding to `NPU {i}`.
```bash
sudo cat /etc/hccn.conf
```

- `xservice_addr` must match the `rpc_server_port` of `xllm_service`

The test command is similar to above. Note that the `PORT` in `curl http://localhost:{PORT}/v1/chat/completions ...` should be the `port` of the prefill node or the `http_server_port` of `xllm service`.