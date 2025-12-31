# Continuous调度器

## 功能介绍
xLLM实现了支持continuous batching的调度策略，continuous_batch是一种动态批处理策略，它不等待批次填满，而是在有请求时就开始处理，同时持续接收新请求并将其加入正在执行的批次中，从而在保持高吞吐量的同时显著降低延迟。

## 使用方式
continuous batching的调度策略在xLLM提供了实现，如果不开其它调度策略，则默认使用continuous batching。

