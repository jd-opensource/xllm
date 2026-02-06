# xTensor Memory Management  

## Background  

Current large model inference engines allocate large contiguous blocks of memory for storing KVCache in one go using a block-based approach. However, this leads to fragmented storage of KVCache and prevents dynamic expansion/shrinking.  

Both GPUs and NPUs provide Virtual Memory Management (VMM) APIs. The VMM API decouples the allocation of virtual and physical memory addresses, allowing physical memory to be mapped to virtual memory on demand. This enables flexible allocation of physical memory while ensuring the continuity of virtual memory.  

Leveraging the VMM API, we have implemented continuous storage for KVCache and on-demand allocation of physical memory. Additionally, we have developed a continuous KVCache version of the Attention operator specifically for the decoding phase.  

## Key Interfaces  

* `PhyPage`: Encapsulation of a physical page.  
* `XTensor`: Encapsulation of virtual memory.  
* `PageAllocator`: Manages the allocation and deallocation of `PhyPage` on a device.  
* `PageManager`: Manages the mapping and unmapping of virtual and physical memory on a device.  
* `PageManagerPool`: Manages all `PageManager` instances across devices.  

## Usage  

Simply add the following gflag parameter when launching xllm:  

```bash  
--enable_continuous_kvcache=true  
```  

## Notes  

Currently, this solution does not support prefix caching, chunked prefill, disaggregated pd, or speculative decoding. These features must be disabled during use:  

```bash  
--enable_prefix_cache=false  
--enable_chunked_prefill=false  
--enable_disagg_pd=false  
--num_speculative_tokens=0  
```  

## Future Work  

* Use the VMM API to unify the management of KVCache and activation values, dynamically adjusting the physical memory usage between them.  
* Use the VMM API to dynamically adjust the KVCache size for multiple LLM models sharing GPUs, enabling efficient workload balancing.
