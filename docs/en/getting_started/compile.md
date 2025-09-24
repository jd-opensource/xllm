# Installation && Compilation

## Container Environment Preparation
First, download the image we provide:
```bash
docker pull quay.io/jd_xllm/xllm-ai:0.6.0-dev-hb-rc2-x86
```
Then create the corresponding container:
```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host  --device=/dev/davinci0  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /var/queue_schedule:/var/queue_schedule -v /mnt/cfs/9n-das-admin/llm_models:/mnt/cfs/9n-das-admin/llm_models -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /usr/local/sbin/:/usr/local/sbin/ -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/:/var/log/npu/slog -v /export/home:/export/home -w /export/home -v ~/.ssh:/root/.ssh  -v /var/log/npu/profiling/:/var/log/npu/profiling -v /var/log/npu/dump/:/var/log/npu/dump -v /home/:/home/  -v /runtime/:/runtime/ quay.io/jd_xllm/xllm-ai:0.6.0-dev-hb-rc2-x86
```

### Docker images

| Device    |    Arch     |   Images      |
|:---------:|:-----------:|:-------------:|
| A2        |     x86     | xllm/xllm-ai:xllm-0.6.0-dev-hb-rc2-py3.11-oe24.03-lts | 
| A2        |     arm     | xllm/xllm-ai:xllm-0.6.0-dev-hb-rc2-py3.11-oe24.03-lts-aarch64 |
| A3        |     arm     | xllm/xllm-ai:xllm-0.6.0-dev-hc-rc2-py3.11-oe24.03-lts-aarch64 |

If you can't download it, you can use the following source instead：

| Device    |    Arch     |   Images      |
|:---------:|:-----------:|:-------------:|
| A2        |     x86     | quay.io/jd_xllm/xllm-ai:0.6.0-dev-hb-rc2-x86|
| A2        |     arm     | quay.io/jd_xllm/xllm-ai:0.6.0-dev-hb-rc2-arm |
| A3        |     arm     | quay.io/jd_xllm/xllm-ai:0.6.0-dev-hc-rc2-arm |

## Installation
After entering the container, download and compile using our [official repository](https://github.com/jd-opensource/xllm):
```bash
git clone https://github.com/jd-opensource/xllm
cd xllm
git submodule init
git submodule update
```
The compilation depends on [vcpkg](https://github.com/microsoft/vcpkg). The Docker image already includes VCPKG_ROOT preconfigured. If you want to manually set it up, you can:
```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && git checkout ffc42e97c866ce9692f5c441394832b86548422c
export VCPKG_ROOT=/your/path/to/vcpkg
```
Then download and install Python dependencies:
```bash
cd xllm
pip install -r cibuild/requirements-dev.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install --upgrade setuptools wheel
```

## Compilation
Execute the compilation to generate the executable file `build/xllm/core/server/xllm` under `build/`. The default device is A2, for A3, add `--device a3`, for mlu, add `--device mlu`:
```bash
python setup.py build
```
Alternatively, you can directly use the following command to compile && generate the whl package under `dist/`:
```bash
python setup.py bdist_wheel
```
