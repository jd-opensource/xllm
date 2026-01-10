### How to compile xllm dynamic library
Run the following command in root directory:
```
python setup.py build --generate-so true
```

If you want to debug, it needs to set DEBUG environment variable.
```
export DEBUG=1
```

### How to install dynamic library
Run installation script xllm/c_api/install.sh, headers and dynamic library will be installed in /usr/local/xllm directory.
```
cd xllm/c_api/tools

sh install.sh
```

You will see the following files in /usr/local/xllm directory:
```
[root@A03-R40-I189-101-4100046]# tree /usr/local/xllm
/usr/local/xllm
|-- include
|   |-- llm.h
|   |-- default.h
|   `-- types.h
`-- lib
    `-- libxllm.so

3 directories, 4 files
```

### How to compile c_api examples
```
cd xllm/c_api/examples
g++ simple_llm_chat_completions.cpp -o simple_llm_chat_completions -I/usr/local/xllm/include -L/usr/local/xllm/lib -lxllm -Wl,-rpath=/usr/local/xllm/lib

```