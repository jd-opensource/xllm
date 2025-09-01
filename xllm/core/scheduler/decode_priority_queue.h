#pragma once
#include <functional>
#include <memory>
#include <queue>
#include <set>

#include "framework/request/priority_comparator.h"
#include "framework/request/request.h"

namespace xllm {

// for Encapsulate and support Iterator pattern
class BaseIterator {
 public:
  virtual ~BaseIterator() = default;
  virtual std::shared_ptr<Request> operator*() const = 0;
  virtual void operator++() = 0;
  virtual bool operator!=(const BaseIterator& other) const = 0;
  virtual std::unique_ptr<BaseIterator> clone() const = 0;
};

template <typename Iterator>
class ConcreteIterator : public BaseIterator {
  Iterator iter_;

 public:
  explicit ConcreteIterator(Iterator iter) : iter_(iter) {}

  std::shared_ptr<Request> operator*() const override { return *iter_; }

  void operator++() override { ++iter_; }

  bool operator!=(const BaseIterator& other) const override {
    const auto* derived = dynamic_cast<const ConcreteIterator*>(&other);
    return derived && iter_ != derived->iter_;
  }

  std::unique_ptr<BaseIterator> clone() const override {
    return std::make_unique<ConcreteIterator>(iter_);
  }
};
class DecodePriorityQueue {
 public:
  class Iterator {
    std::unique_ptr<BaseIterator> itr_;

   public:
    explicit Iterator(std::unique_ptr<BaseIterator> itr)
        : itr_(std::move(itr)) {}

    std::shared_ptr<Request> operator*() const { return **itr_; }

    Iterator& operator++() {
      ++*itr_;
      return *this;
    }

    bool operator!=(const Iterator& other) const {
      return itr_->operator!=(*other.itr_);
    }
  };
  virtual void push(std::shared_ptr<Request> req) = 0;
  virtual void push(std::shared_ptr<Request> req, bool if_back) = 0;
  virtual void pop_top() = 0;
  virtual void pop_back() = 0;
  virtual std::shared_ptr<Request> top() const = 0;
  virtual std::shared_ptr<Request> back() const = 0;
  virtual bool empty() const = 0;
  virtual size_t size() const = 0;
  virtual ~DecodePriorityQueue() = default;

  virtual Iterator begin() const = 0;
  virtual Iterator end() const = 0;
  virtual Iterator rbegin() const = 0;
  virtual Iterator rend() const = 0;
};

class DynamicPriorityQueue : public DecodePriorityQueue {
 private:
  using QueueType =
      std::set<std::shared_ptr<Request>,
               std::function<bool(const std::shared_ptr<Request>&,
                                  const std::shared_ptr<Request>&)>>;
  QueueType queue_;
  std::unique_ptr<PriorityComparator> comparator_;

 public:
  explicit DynamicPriorityQueue(std::unique_ptr<PriorityComparator> comparator)
      : comparator_(std::move(comparator)),
        queue_([this](const auto& a, const auto& b) {
          return !(*comparator_)(a, b);  // assign to Priority Comparator
        }) {}

  void push(std::shared_ptr<Request> req) override { queue_.insert(req); }
  void push(std::shared_ptr<Request> req, bool if_back) override {
    LOG(FATAL) << "DynamicPriorityQueue not support";
  }
  void pop_top() override { queue_.erase(queue_.begin()); }
  void pop_back() override { queue_.erase(std::prev(queue_.end())); }
  std::shared_ptr<Request> top() const override { return *queue_.begin(); }
  std::shared_ptr<Request> back() const override { return *queue_.rbegin(); }
  bool empty() const override { return queue_.empty(); }
  virtual size_t size() const override { return queue_.size(); }

  Iterator begin() const override {
    return Iterator(std::make_unique<ConcreteIterator<QueueType::iterator>>(
        queue_.begin()));
  }

  Iterator end() const override {
    return Iterator(
        std::make_unique<ConcreteIterator<QueueType::iterator>>(queue_.end()));
  }

  Iterator rbegin() const override {
    return Iterator(
        std::make_unique<ConcreteIterator<QueueType::reverse_iterator>>(
            queue_.rbegin()));
  }

  Iterator rend() const override {
    return Iterator(
        std::make_unique<ConcreteIterator<QueueType::reverse_iterator>>(
            queue_.rend()));
  }
};

class FCFSQueue : public DecodePriorityQueue {
  // use deque to implement FCFS queue for insert and evict effeciency
 private:
  std::deque<std::shared_ptr<Request>> queue_;

 public:
  void push(std::shared_ptr<Request> req) override { queue_.push_front(req); }

  void push(std::shared_ptr<Request> req, bool if_back) override {
    if (if_back) {
      queue_.push_back(req);
    } else {
      queue_.push_front(req);
    }
  }

  void pop_top() override { queue_.pop_front(); }
  void pop_back() override { queue_.pop_back(); }
  std::shared_ptr<Request> top() const override { return queue_.front(); }
  std::shared_ptr<Request> back() const override { return queue_.back(); }
  bool empty() const override { return queue_.empty(); }
  virtual size_t size() const override { return queue_.size(); }

  Iterator begin() const override {
    return Iterator(
        std::make_unique<ConcreteIterator<decltype(queue_)::const_iterator>>(
            queue_.begin()));
  }

  Iterator end() const override {
    return Iterator(
        std::make_unique<ConcreteIterator<decltype(queue_)::const_iterator>>(
            queue_.end()));
  }

  Iterator rbegin() const override {
    return Iterator(std::make_unique<
                    ConcreteIterator<decltype(queue_)::const_reverse_iterator>>(
        queue_.rbegin()));
  }

  Iterator rend() const override {
    return Iterator(std::make_unique<
                    ConcreteIterator<decltype(queue_)::const_reverse_iterator>>(
        queue_.rend()));
  }
};
}  // namespace xllm