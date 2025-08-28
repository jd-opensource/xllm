# Prefix Cache Optimization

## Feature Introduction
xLLM supports prefix cache matching. The prefix cache is based on `murmur_hash` and uses an LRU eviction policy, delivering superior matching efficiency and increased prefix cache hit rates.
Additionally, the prefix cache has been optimized to support the `continuous_scheduler`, `chunked_scheduler`, and `zero_evict_scheduler`. The cache is updated immediately after prefill operations, enhancing matching timeliness. For the `chunked_scheduler`, multi-stage chunked prefill matching is supported, reducing computational overhead and minimizing KV cache usage as much as possible.

## Usage
The prefix cache is implemented in xLLM and exposed through gflags parameters to control its functionality.

- Enable prefix cache with specific policy and settings:
```
--prefix_cache_policy=murmur_hash3
--enable_prefix_cache=true
```

## Performance Impact
After enabling prefix cache, on the Qwen3-8B model with a TPOT constraint of 50ms, the E2E latency **decreased by 10%**.

!!! warning "Note"
    PD separation scheduler is not currently supported.