# Topk&Topp算子优化

## 背景
在自然语言生成任务中，topK和topP采样策略被广泛应用于控制生成文本的多样性和质量。然而，在小模型中，这两种策略的计算耗时相对较长。这主要是由于小模型的参数较少，导致在处理概率分布时，排序和筛选的效率降低，从而影响了生成速度。因此，优化小模型中topK和topP的实现，可以提升其采样效率。


## 功能介绍

topKtopP算子的实现将排序、topK、softmax和topP等多个小算子融合为一个大算子，从而提高了计算效率和性能。


## 用户接口
### 算子调用API
```c++
void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP);
```

- `logits`: 输入的logits张量，包含模型的输出分数。
- `topK`: 用于选择的前K个概率的阈值张量。
- `topP`: 用于选择的累积概率的阈值张量。


## 性能效果

* 使用topKtopP融合算子后，在qwen2-0.5B模型中，TTOT **下降37%**,TTFT **提升10%**。
