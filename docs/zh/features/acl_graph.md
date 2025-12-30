# ACLGraph


## 功能介绍

为了优化Host侧调度性能，NPU近期推出了类似CUDA Graph的图模式方案ACLGraph。与采用CPU密集小任务提交、NPU频繁启动小Kernel的传统模式相比，ACLGraph模式通过在CPU一次提交大任务后，NPU内部流式执行小kernel，显著降低了启动时间和NPU气泡。

在xLLM引擎中使用ACLGraph功能，我们实现了以下特性：
### 动态维度参数化
  - 将关键动态维度（如批大小和序列长度）作为整图输入参数，从而提高灵活性。在进行图的内存分配和内核配置时，利用这些动态参数计算实际所需值，例如通过公式   $block\_table\_size = batch\_size \times (max\_seq\_len / block\_size)$ 计算block_table的大小。在图启动阶段，则将实际的批大小和最大序列长度作为参数传入，以确保kerenel能够使用正确的stride来访问数据。

### 多shape复用的显存池
  - 为了避免多shape使用单独显存buffer（输入、输出和中间Tensor）导致浪费，我们采用了可扩张的显存池。多shape复用基地址，不同shape对池基地址的偏移量（Offset）不同。


## 使用方式

上述功能已经在xLLM引擎内部进行了实现，对用户透明，用户无需关注内部实现细节，在适用的场景直接开启相关功能即可。通过gflags参数`enable_graph`开启。参数默认为false，如需开启在xLLM的服务启动脚本中设置为true即可，示例如下：
```shell
--enable_graph=true
```


## 性能效果
- 开启ACLGraph功能后，在Qwen3-0.6B和Qwen3-1.7B等模型上，decode阶段吞吐 **提升8%-10%**。

!!! warning "注意事项"
    - 为新模型添加ACLGraph支持时，需要check计算过程中用到的kerenel是否实现了动态维度参数化。如果没有，需要重新实现kernel。

!!! tip "未来计划"
    * 支持MoE模型Attention DP和FFN EP之间的通信操作适配不同shape。
