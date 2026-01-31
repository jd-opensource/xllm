# Topk & Topp Operator Optimization

## Background

In natural language generation tasks, the topK and topP sampling strategies are widely used to control the diversity and quality of generated text. However, in small models, the computation time for these two strategies is relatively long. This is mainly due to the fewer parameters in small models, which leads to reduced efficiency in sorting and filtering when processing probability distributions, thereby affecting generation speed. Therefore, optimizing the implementation of topK and topP in small models can enhance their sampling efficiency.

## Feature Introduction

The implementation of the topKtopP operator merges multiple small operators, such as sorting, topK, softmax, and topP, into a single large operator, thereby improving computational efficiency and performance.

## User Interface

### Operator Call API

```c++
void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP);
```

- `logits`: The input logits tensor containing the model's output scores.
- `topK`: The threshold tensor for selecting the top K probabilities.
- `topP`: The threshold tensor for selecting the cumulative probabilities.

## Performance Effect

* After using the topKtopP fused operator, in the qwen2-0.5B model, TTOT **decreased by 37%**, and TTFT **increased by 10%**.