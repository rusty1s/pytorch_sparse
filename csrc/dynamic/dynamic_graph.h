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
    cols.assign(col_data, col_data + cnt);
    if (val) {
      V *val_data = reinterpret_cast<V *>(val->data_ptr<V>());
      vals->assign(val_data, val_data + cnt);
    }
  }

  void erase_val(int64_t col) {
    for (std::size_t i = 0; i < cols.size(); i++) {
      if (cols[i] == col) {
        cols[i] = cols.back();
        cols.pop_back();
        (*vals)[i] = vals->back();
        vals->pop_back();
        return;
      }
    }
  }

  void erase(int64_t col) {
    for (std::size_t i = 0; i < cols.size(); i++) {
      if (cols[i] == col) {
        cols[i] = cols.back();
        cols.pop_back();
        return;
      }
    }
  }

  void insert_val(int64_t col, V val) {
    cols.push_back(col);
    vals->push_back(val);
  }

  void insert(int64_t col) { cols.push_back(col); }

  std::vector<int64_t> cols;
  std::optional<std::vector<V>> vals;
};

template <typename V> class DynamicGraph {
public:
  bool has_node(int64_t node) { return rows_.count(node); }

  int64_t *colptr(int64_t node) {
    return rows_[node].cols.data();
  }

  size_t degree(int64_t node) {
    return rows_[node].cols.size();
  }

  V *valptr(int64_t node) {
    return rows_[node].vals->data();
  }

  void update(DynamicLog<V> log, torch::Tensor rowptr, torch::Tensor col,
              std::optional<torch::Tensor> val) {
    if (!rows_.count(log.src)) {
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
