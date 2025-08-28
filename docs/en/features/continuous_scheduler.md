# Continuous Scheduler

## Feature Introduction
xLLM implements a scheduling strategy that supports continuous batching. Continuous batching is a dynamic batching strategy that does not wait for a batch to be filled. Instead, it starts processing as soon as requests are available, while continuously accepting new requests and adding them to the currently executing batch. This approach significantly reduces latency while maintaining high throughput.

## Usage
The continuous batching scheduling strategy is implemented in xLLM. If no other scheduling strategies are enabled, continuous batching is used by default.