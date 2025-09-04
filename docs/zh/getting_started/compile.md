
# 安装编译

## 容器环境准备
首先下载我们提供的镜像：
```bash
docker pull xllm/xllm-ai:xllm-0.6.0-dev-hb-py3.11-oe24.03-lts
```
然后创建对应的容器
```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host  --device=/dev/davinci0  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /var/queue_schedule:/var/queue_schedule -v /mnt/cfs/9n-das-admin/llm_models:/mnt/cfs/9n-das-admin/llm_models -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /usr/local/sbin/:/usr/local/sbin/ -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/:/var/log/npu/slog -v /export/home:/export/home -w /export/home -v ~/.ssh:/root/.ssh  -v /var/log/npu/profiling/:/var/log/npu/profiling -v /var/log/npu/dump/:/var/log/npu/dump -v /home/:/home/  -v /runtime/:/runtime/  xllm/xllm-ai:xllm-0.6.0-dev-hb-py3.11-oe24.03-lts
```

### 镜像列表

| Device    |    Arch     |   Images      |
|:---------:|:-----------:|:-------------:|
| A2        |     x86     | xllm/xllm-ai:xllm-0.6.0-dev-hb-py3.11-oe24.03-lts | 
| A2        |     arm     | xllm/xllm-ai:xllm-0.6.0-dev-hb-py3.11-oe24.03-lts-aarch64 |
| A3        |     arm     | xllm/xllm-ai:xllm-0.6.0-dev-hc-py3.11-oe24.03-lts-aarch64 |


## 安装
进入到容器后，使用我们的[官方仓库](https://github.com/jd-opensource/xllm)下载编译：
```bash
git clone https://github.com/jd-opensource/xllm
cd xllm 
git submodule init
git submodule update
```
编译依赖[vcpkg](https://github.com/microsoft/vcpkg)，镜像中已经提前配置完成。如果您想要手动配置，可以执行如下命令:
```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && git checkout ffc42e97c866ce9692f5c441394832b86548422c
export VCPKG_ROOT=/your/path/to/vcpkg
```
然后下载安装python依赖:
```
cd xllm
pip install -r cibuild/requirements-dev.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install --upgrade setuptools wheel
```
## 编译
执行编译，在`build/`下生成可执行文件`build/xllm/core/server/xllm`：
```bash
python setup.py build
```
也可以直接用以下命令编译&&在`dist/`下生成whl包: 
```bash
python setup.py bdist_wheel
```
