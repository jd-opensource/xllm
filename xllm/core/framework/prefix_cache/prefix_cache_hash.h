/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <atomic>
#include <string>
#include <string_view>
#include <unordered_map>

#include "prefix_cache/prefix_cache.h"
#include "util/double_buffer.h"
#include "util/threadpool.h"

namespace xllm {

using DoubleBufferKvCacheEvent = DoubleBuffer<KvCacheEvent>;

class PrefixCacheHash : public PrefixCache {
 public:
  explicit PrefixCacheHash(uint32_t block_size)
      : block_size_(block_size), num_blocks_(0) {};

  ~PrefixCacheHash() {
    exited_.store(true);
    sleep(2);
  };

  virtual KvCacheEvent* get_upload_kvcache_events() override {
    LOG(ERROR) << "Not implemented!";
    return nullptr;
  }

 protected:
  struct Node {
    Block block;
    // the last access time of the node, used to evict blocks
    int64_t last_access_time = 0;

    // the previous and next nodes, used to maintain the LRU list
    Node* prev = nullptr;
    Node* next = nullptr;
  };

  struct DNodeList {
    DNodeList() {
      lst_front.next = &lst_back;
      lst_back.prev = &lst_front;
    }

    ~DNodeList() {
      Node* node = lst_front.next;
      while (node != &lst_back) {
        Node* next = node->next;
        delete node;
        node = next;
      }
    }

    bool is_empty() { return lst_front.next == &lst_back; }

    // remove the node from the LRU list, and return next node
    Node* remove_node(Node* node) {
      Node* next_node = node->next;

      node->prev->next = next_node;
      next_node->prev = node->prev;

      return next_node;
    }

    bool is_last(Node* node) { return node == &lst_back; }

    // add a new node to the front of the LRU list
    void push_front(Node* node) {
      node->next = lst_front.next;
      lst_front.next->prev = node;

      node->prev = &lst_front;
      lst_front.next = node;
    }

    Node* get_first() { return lst_front.next; }

    // pop out node to the back of the LRU list
    Node* pop_front() {
      if (lst_front.next == &lst_back) {
        return nullptr;
      }

      Node* node = lst_front.next;

      lst_front.next = node->next;
      node->next->prev = &lst_front;

      return node;
    }

    // add a new node to the back of the LRU list
    void push_back(Node* node) {
      node->prev = lst_back.prev;
      node->next = &lst_back;
      lst_back.prev->next = node;
      lst_back.prev = node;
    }

    // move the node to the back of the LRU list
    void move_back(Node* node) {
      remove_node(node);
      push_back(node);
    }

    // Node lst_front;
    Node lst_front;
    Node lst_back;
  };

  DNodeList lru_lst_;

  // the block size of the memory blocks
  uint32_t block_size_;

  // the total number of blocks in the prefix cache
  size_t num_blocks_ = 0;

  std::atomic_bool exited_{false};
};

}  // namespace xllm
