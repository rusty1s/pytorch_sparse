#pragma once

#include "dynamic_graph.h"
#include "dynamic_log.h"
#include <algorithm>
#include <optional>
#include <shared_mutex>
#include <stdlib.h>
#include <time.h>
#include <unordered_map>
#include <vector>

template <typename V> struct RowSampleResult {
  RowSampleResult() {}
  std::vector<int64_t> neighbors;
  std::optional<std::vector<V>> vals;
};

template <typename V>
std::vector<RowSampleResult<V>> sample_row(int64_t *colptr, size_t deg,
                                           V *valptr, int64_t num_neighbors,
                                           bool replace) {
  std::vector<RowSampleResult<V>> ret;
  return ret;
}

template <typename V> class DynamicBlock {
public:
  DynamicBlock(torch::Tensor rowptr, torch::Tensor col,
               std::optional<torch::Tensor> val)
      : rowptr_(rowptr), col_(col), val_(val), mu_(new std::shared_mutex()) {}

  void apply_log_locked(int64_t src, int64_t dst, std::optional<V> val,
                        bool insert) {
    graph_.update(src, dst, val, insert, rowptr_, col_, val_);
  }

  std::vector<RowSampleResult<V>> sample_adj_locked(std::vector<int64_t> idx,
                                                    int64_t num_neighbors,
                                                    bool replace) {
    std::vector<RowSampleResult<V>> ret;
    int64_t *colptr = nullptr;
    V *valptr = nullptr;
    size_t deg = 0;
    int64_t *rowptr_data = rowptr_.data_ptr<int64_t>();
    int64_t *col_data = col_.data_ptr<int64_t>();
    V *val_data = val_ ? val_->data_ptr<V>() : nullptr;
    for (auto node : idx) {
      if (graph_.has_node(node)) {
        colptr = graph_.colptr(node);
        deg = graph_.degree(node);
        if (val_) {
          valptr = graph_.valptr(node);
        }
      } else {
        int64_t base = rowptr_data[node];
        colptr = col_data + base;
        deg = rowptr_data[node + 1] - base;
        if (val_) {
          valptr = val_data + base;
        }
      }
      ret.emplace_back(sample_row<V>(colptr, deg, valptr));
    }
    return ret;
  }

  std::shared_ptr<std::shared_mutex> get_mutex() { return mu_; }

private:
  torch::Tensor rowptr_;
  torch::Tensor col_;
  std::optional<torch::Tensor> val_;
  DynamicGraph<V> graph_;
  std::shared_ptr<std::shared_mutex> mu_;
};

template <typename V> class DynamicStore {
public:
  DynamicStore(std::size_t bn, std::size_t bs, std::size_t nn, std::size_t cap)
      : block_num_(bn), block_size_(bs), node_num_(nn), logstore_(cap) {}

  void append_block(torch::Tensor rowptr, torch::Tensor col,
                    std::optional<torch::Tensor> val) {
    blocks_.emplace_back(rowptr, col, val);
  }

  void apply_log(std::unique_ptr<std::vector<DynamicLog<V>>> logs) {
    std::unordered_map<std::size_t, std::vector<std::size_t>> m;
    for (std::size_t i = 0; i < logs->size(); i++) {
      std::size_t block_id = get_block((*logs)[i].src);
      m[block_id].push_back(i);
    }
    while (!m.empty()) {
      auto p = m.begin();
      auto mu = blocks_[p->first].get_mutex();
      {
        std::unique_lock<std::shared_mutex> lock(*mu);
        for (auto i : p->second) {
          blocks_[p->first].apply_log_locked(get_offset((*logs)[i].src),
                                             (*logs)[i].dst, (*logs)[i].val,
                                             (*logs)[i].insert);
        }
      }
      m.erase(p);
    }
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
             std::optional<torch::Tensor>>
  sample_adj(torch::Tensor idx, int64_t num_neighbors, bool replace);
  // srand(time(NULL));

private:
  std::size_t block_num_;
  std::size_t block_size_;
  std::size_t node_num_;
  std::vector<DynamicBlock<V>> blocks_;
  DynamicLogStore<V> logstore_;

  std::size_t get_block(int64_t node) { return node / block_size_; }

  std::size_t get_offset(int64_t node) { return node % block_size_; }

  int64_t get_global(int64_t node, size_t block_id) {
    return block_id * block_size_ + node;
  }
};
