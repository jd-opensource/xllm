# xTensor显存管理

## 背景介绍

目前的大模型推理引擎都是使用基于 block 的方式一次性分配大块连续显存用于存储 KVCache，然而这会造成 KVCache 的离散存储，且无法动态扩容/缩容。

而 GPU 和 NPU 都提供了虚拟内存管理 API（Virtual Memory Management，VMM），VMM API 可以将显存的虚拟地址和物理地址的分配解耦，然后将物理内存按需映射到虚拟内存上，从而实现物理内存的弹性分配，并保证虚拟内存的连续性。

基于 VMM API，我们实现了 KVCache 的连续存储及按需分配物理内存，并且实现了针对解码阶段的连续 KVCache 版本的 Attention 算子。

## 主要接口
* `PhyPage`：对物理页的封装。
* `XTensor`：对虚拟内存的封装。
* `PageAllocator`：用于管理一个device上的`PhyPage`的分配与回收。
* `PageManager`：用于管理一个device上虚拟内存与物理内存的映射与取消映射。
* `PageManagerPool`：用于管理所有的device上的`PageManager`。

## 使用方式
只需在启动 xLLM 时加上下面的 gflag 参数即可：

```bash
--enable_continuous_kvcache=true
```

!!! warning "注意事项"
    目前该方案暂不支持prefix cacheing，chunked prefill，disaggregated pd，speculative decoding，在使用时需要将这些功能关闭：
    ```bash
    --enable_prefix_cache=false
    --enable_chunked_prefill=false
    --enable_disagg_pd=false
    --num_speculative_tokens=0
    ```

!!! tip "未来计划"
    * 使用 VMM API 将 KVCache 和激活值统一管理，并动态管理二者使用的物理显存大小。
    * 使用 VMM API 实现当多个 LLM 模型共享 GPUs时，动态调整它们使用的 KVCache 的大小从而实现高效负载。

