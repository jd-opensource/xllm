#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {
// a singleton mode by version
template <typename T>
class VersionSingleton {
 public:
  template <typename... Args>
  static T* GetInstance(const std::string& version,
                        bool delete_old_versions = true,
                        int reserved_version_size =
                            2,  // default retention of the last two versions
                        Args&&... args) {
    T* instance = nullptr;

    {
      std::lock_guard<std::mutex> lock(instance_map_mutex_);
      auto it = instance_map_.find(version);
      if (it != instance_map_.end()) {
        instance = it->second.get();
      }
    }

    if (instance == nullptr) {
      std::lock_guard<std::mutex> lock(instance_map_mutex_);

      auto it = instance_map_.find(version);
      if (it == instance_map_.end()) {
        instance = new T(std::forward<Args>(args)...);
        instance_map_[version] = std::unique_ptr<T>(instance);
        instance_version_list_.push_front(version);
        if (delete_old_versions) {
          if (instance_version_list_.size() > reserved_version_size) {
            auto it = instance_version_list_.begin();
            std::advance(it, reserved_version_size);
            for (; it != instance_version_list_.end(); it++) {
              instance_map_.erase(*it);
            }
            instance_version_list_.resize(reserved_version_size);
          }
        }
      } else {
        instance = it->second.get();
      }
    }

    return instance;
  }

  static std::vector<std::string> GetVersions() {
    std::lock_guard<std::mutex> lock(instance_map_mutex_);
    std::vector<std::string> versions;
    for (const auto& pair : instance_map_) {
      versions.push_back(pair.first);
    }
    return versions;
  }

  static void DestroyInstance(const std::string& version) {
    std::lock_guard<std::mutex> lock(instance_map_mutex_);
    instance_map_.erase(version);
  }

  static void DestroyAllInstances() {
    std::lock_guard<std::mutex> lock(instance_map_mutex_);
    instance_map_.clear();
  }

  VersionSingleton(const VersionSingleton&) = delete;
  VersionSingleton& operator=(const VersionSingleton&) = delete;

 private:
  VersionSingleton() = default;
  ~VersionSingleton() = default;

  static std::unordered_map<std::string, std::unique_ptr<T>> instance_map_;
  static std::list<std::string> instance_version_list_;
  static std::mutex instance_map_mutex_;
};

template <typename T>
std::unordered_map<std::string, std::unique_ptr<T>>
    VersionSingleton<T>::instance_map_;
template <typename T>
std::list<std::string> VersionSingleton<T>::instance_version_list_;
template <typename T>
std::mutex VersionSingleton<T>::instance_map_mutex_;

}  // namespace xllm