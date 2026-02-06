# Prefix Cache 优化

## 功能介绍
xLLM支持prefix_cache匹配。prefix_cache基于mermer_hash，使用lru淘汰策略，提供更极致的匹配效率，同时提高prefix_cache命中率。
同时对prefix_cache进行了优化，支持continuous_scheduler、chunked_scheduler和zero_evict_scheduler，在prefill之后即更新
prefix_cache，提高匹配时效性，同时对于chunked_scheduler，支持多阶段chunked_prefill匹配，减少计算量并尽可能减少kv_cache占用。

## 使用方式
prefix_cache已在xLLM实现，并向外暴露gflag参数，控制功能的开关。

- 开启zero_evict策略，并设置max_decode_token_per_sequence。
```
--enable_prefix_cache=true
```

## 性能效果
开启prefix_cache之后，在Qwen3-8B模型上，限制TPOT50ms，E2E时延 **下降10%**。

!!! warning "注意"
    暂不支持PD分离调度器