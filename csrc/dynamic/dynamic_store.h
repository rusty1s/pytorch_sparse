#pragma once

#include "dynamic_graph.h"
#include "dynamic_log.h"
#include <optional>
#include <vector>

template <typename V> class DynamicBlock {
public:
  DynamicBlock(torch::Tensor rowptr, torch::Tensor col,
               std::optional<torch::Tensor> val)
      : rowptr_(rowptr), col_(col), val_(val) {}

  void apply_log(int64_t src, int64_t dst, std::optional<V> val, bool insert) {
    graph_.update(src, dst, val, insert, rowptr_, col_, val_);
  }

private:
  torch::Tensor rowptr_;
  torch::Tensor col_;
  std::optional<torch::Tensor> val_;
  DynamicGraph<V> graph_;
};

template <typename V> class DynamicStore {
public:
  DynamicStore(std::size_t bn, std::size_t bs, std::size_t nn)
      : block_num_(bn), block_size_(bs), node_num_(nn) {}

  void append_block(torch::Tensor rowptr, torch::Tensor col,
                    std::optional<torch::Tensor> val) {
    blocks_.emplace_back(rowptr, col, val);
  }

  void apply_log(int64_t src, int64_t dst, std::optional<V> val, bool insert) {
    std::size_t block_id = src / block_size_;
    src = src % block_size_;
    blocks_[block_id].apply_log(src, dst, val, insert);
  }

private:
  std::size_t block_num_;
  std::size_t block_size_;
  std::size_t node_num_;
  std::vector<DynamicBlock<V>> blocks_;
};
