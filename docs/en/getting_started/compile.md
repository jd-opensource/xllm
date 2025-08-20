# Installation && Compilation

## Container Environment Preparation
First, download the image we provide:
```bash
docker pull xllm-ai/xllm:0.6.0-dev-800I-A3-py3.11-openeuler24.03-lts-aarch64
```
Then create the corresponding container:
```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host  --device=/dev/davinci0  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /var/queue_schedule:/var/queue_schedule -v /mnt/cfs/9n-das-admin/llm_models:/mnt/cfs/9n-das-admin/llm_models -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /usr/local/sbin/:/usr/local/sbin/ -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/:/var/log/npu/slog -v /export/home:/export/home -w /export/home -v ~/.ssh:/root/.ssh  -v /var/log/npu/profiling/:/var/log/npu/profiling -v /var/log/npu/dump/:/var/log/npu/dump -v /home/:/home/  -v /runtime/:/runtime/  xllm-ai:xllm-0.6.0-dev-800I-A3-py3.11-openeuler24.03-lts-aarch64
```

## Installation
After entering the container, download and compile using our [official repository](https://github.com/jd-opensource/xllm):
```bash
git clone https://github.com/jd-opensource/xllm
cd xllm
git submodule init
git submodule update
```
The compilation depends on [vcpkg](https://github.com/microsoft/vcpkg). `vcpkg` will be downloaded by default during compilation. You can also download `vcpkg` in advance and set the environment variable:
```bash
git clone https://github.com/microsoft/vcpkg.git
export VCPKG_ROOT=/your/path/to/vcpkg
```
Then download and install Python dependencies:
```bash
cd xllm
pip install -r cibuild/requirements-dev.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install --upgrade setuptools wheel
```

## Compilation
Execute the compilation to generate the executable file `build/xllm/core/server/xllm` under `build/`:
```bash
python setup.py build
```
Alternatively, you can directly use the following command to compile && generate the whl package under `dist/`:
```bash
python setup.py bdist_wheel
```