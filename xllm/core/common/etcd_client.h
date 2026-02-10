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

#include <etcd/KeepAlive.hpp>
#include <etcd/SyncClient.hpp>
#include <etcd/Watcher.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

using Callback = std::function<void(const etcd::Response&)>;

class EtcdClient {
 public:
  EtcdClient(const std::string& etcd_addr);
  ~EtcdClient();

  void add_watch(const std::string& key_prefix,
                 Callback callback,
                 bool recursive = true);

  void remove_watch(const std::string& key_prefix);

  void stop_watch();

  bool register_instance(const std::string& key,
                         const std::string& value,
                         const int ttl);

  bool get_master_service(const std::string& key_prefix, std::string* values);

  bool get_all_xservices(const std::string& key_prefix,
                         std::vector<std::string>* values);

 private:
  struct WatcherInfo {
    std::unique_ptr<etcd::Watcher> watcher;
    Callback callback;
  };

  etcd::SyncClient client_;
  std::string etcd_addr_;
  std::mutex watchers_mutex_;
  std::unordered_map<std::string, WatcherInfo> watchers_;
  std::vector<std::shared_ptr<etcd::KeepAlive>> keep_alives_;
};

}  // namespace xllm
