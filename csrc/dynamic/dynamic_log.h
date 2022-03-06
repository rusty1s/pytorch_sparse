#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

template <typename V> class DynamicLog {
public:
  DynamicLog(int64_t s, int64_t d, std::optional<V> v, bool ins)
      : src(s), dst(d), val(v), insert(ins) {}

  int64_t src;
  int64_t dst;
  std::optional<V> val;
  bool insert;
};

template <typename V> class DynamicLogStore {
public:
  DynamicLogStore(int capacity)
      : cap_(capacity), immutable_(nullptr), mutable_(nullptr) {}

  std::unique_ptr<std::vector<DynamicLog<V>>> get_immutable() {
    std::lock_guard<std::mutex> lock(mu_);
    return std::move(immutable_);
  }

  bool apply_log(int64_t src, int64_t dst, std::optional<V> val, bool insert) {
    if (mutable_ && mutable_->size() >= cap_) {
      std::lock_guard<std::mutex> lock(mu_);
      if (immutable_ == nullptr) {
        std::swap(mutable_, immutable_);
      } else {
        return false;
      }
    }
    if (mutable_ == nullptr) {
      mutable_.reset(new std::vector<DynamicLog<V>>());
    }
    mutable_->emplace_back(src, dst, val, insert);
    return true;
  }

private:
  std::size_t cap_;
  std::unique_ptr<std::vector<DynamicLog<V>>> immutable_;
  std::unique_ptr<std::vector<DynamicLog<V>>> mutable_;
  std::mutex mu_;
};
