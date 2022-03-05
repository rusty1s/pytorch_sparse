#pragma once

#include "dynamic_log.h"
#include <optional>
#include <torch/extension.h>
#include <tuple>
#include <unordered_map>
#include <vector>

template <typename V> class DynamicRow {
public:
  DynamicRow(int64_t node, torch::Tensor rowptr, torch::Tensor col,
             std::optional<torch::Tensor> val) {
    int64_t *rowptr_data =
        reinterpret_cast<int64_t *>(rowptr.data_ptr<int64_t>());
    int64_t *col_data = reinterpret_cast<int64_t *>(col.data_ptr<int64_t>());
    std::size_t cnt = rowptr_data[node + 1] - rowptr_data[node];
    cols_.assign(col_data, col_data + cnt);
    if (val) {
      V *val_data = reinterpret_cast<V *>(val->data_ptr<V>());
      vals_->assign(val_data, val_data + cnt);
    }
  }

  void erase_val(int64_t col) {
    for (std::size_t i = 0; i < cols_.size(); i++) {
      if (cols_[i] == col) {
        cols_[i] = cols_.back();
        cols_.pop_back();
        (*vals_)[i] = vals_->back();
        vals_->pop_back();
        return;
      }
    }
  }

  void erase(int64_t col) {
    for (std::size_t i = 0; i < cols_.size(); i++) {
      if (cols_[i] == col) {
        cols_[i] = cols_.back();
        cols_.pop_back();
        return;
      }
    }
  }

  void insert_val(int64_t col, V val) {
    cols_.push_back(col);
    vals_->push_back(*val);
  }

  void insert(int64_t col) { cols_.push_back(col); }

private:
  std::vector<int64_t> cols_;
  std::optional<std::vector<V>> vals_;
};

template <typename V> class DynamicGraph {
public:
  DynamicGraph() {}

  void update(DynamicLog<V> log, torch::Tensor rowptr, torch::Tensor col,
              std::optional<torch::Tensor> val) {
    if (!rows.count(log.src)) {
      rows_[log.src] = DynamicRow(log.src, rowptr, col, val);
    }
    if (log.insert) {
      if (log.val) {
        rows_[log.src].insert_val(log.col, *log.val);
      } else {
        rows_[log.src].insert(log.col);
      }
    } else {
      if (log.val) {
        rows_[log.src].erase_val(log.col);
      } else {
        rows_[log.src].erase(log.col);
      }
    }
  }

private:
  std::unordered_map<int64_t, DynamicRow<V>> rows_;
};
