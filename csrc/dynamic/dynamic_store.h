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
RowSampleResult<V> sample_row(int64_t *colptr, std::size_t deg, V *valptr,
                              int64_t num_neighbors, bool replace) {
  RowSampleResult<V> ret;
  if (valptr) {
    ret.vals = std::vector<V>();
  }
  if (num_neighbors < 0 ||
      !replace &&
          deg <= num_neighbors) { // No sampling
                                  // ======================================

    for (std::size_t j = 0; j < deg; j++) {
      ret.neighbors.push_back(colptr[j]);
      if (valptr) {
        ret.vals->push_back(valptr[j]);
      }
    }
  }

  else if (replace) { // Sample with replacement ===============================

    for (int64_t j = 0; j < num_neighbors; j++) {
      ret.neighbors.push_back(colptr[j]);
      if (valptr) {
        ret.vals->push_back(valptr[j]);
      }
    }
  } else { // Sample without replacement via Robert Floyd algorithm ============

    std::unordered_set<int64_t> perm;
    for (int64_t j = deg - num_neighbors; j < deg; j++) {
      if (!perm.insert(rand() % j).second) {
        perm.insert(j);
      }
    }

    for (const int64_t &p : perm) {
      ret.neighbors.push_back(colptr[p]);
      if (valptr) {
        ret.vals->push_back(valptr[p]);
      }
    }
  }
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
    std::size_t deg = 0;
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
      : block_num_(bn), block_size_(bs), node_num_(nn), logstore_(cap),
        has_val_(false) {}

  void append_block(torch::Tensor rowptr, torch::Tensor col,
                    std::optional<torch::Tensor> val) {
    if (val) {
      has_val_ = true;
    }
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
      {
        auto mu = blocks_[p->first].get_mutex();
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
  sample_adj(torch::Tensor idx, int64_t num_neighbors, bool replace) {
    srand(time(NULL));
    auto idx_data = idx.data_ptr<int64_t>();
    auto out_rowptr = torch::empty(idx.numel() + 1, idx.options());
    auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
    out_rowptr_data[0] = 0;

    std::vector<std::vector<std::tuple<int64_t, int64_t>>> cols; // col, e_id
    std::vector<int64_t> n_ids;
    std::unordered_map<int64_t, int64_t> n_id_map;

    int64_t i;
    for (int64_t n = 0; n < idx.numel(); n++) {
      i = idx_data[n];
      cols.push_back(std::vector<std::tuple<int64_t, V>>());
      n_id_map[i] = n;
      n_ids.push_back(i);
    }

    std::unordered_map<std::size_t, std::vector<std::size_t>> m;
    for (std::size_t i = 0; i < idx.numel(); i++) {
      auto node = idx_data[i];
      auto block_id = get_block(node);
      m[block_id].push_back(i);
    }

    while (!m.empty()) {
      for (auto &p : m) {
        auto &[block_id, idxs] = p;
        auto mu = blocks_[block_id].get_mutex();
        std::vector<int64_t> nodes;
        for (auto i : idxs) {
          nodes.push_back(get_offset(idx_data[i]));
        }
        if (mu->try_lock_shared()) {
          // TODO: lock guard
          auto res = blocks_[block_id].sample_adj_locked(
              std::move(nodes), num_neighbors, replace);
          mu->unlock();
          m.erase(block_id);
          for (int i = 0; i < nodes.size(); i++) {
            auto node = get_global(nodes[i], block_id);
            auto &[neighbors, vals] = res[i];
            out_rowptr_data[idxs[i] + 1] = neighbors.size();
            for (int j = 0; j < neighbors.size(); j++) {
              auto c = neighbors[j];
              V v = has_val_ ? (*vals)[j] : 0;
              if (n_id_map.count(c) == 0) {
                n_id_map[c] = n_ids.size();
                n_ids.push_back(c);
              }
              cols[idxs[i]].push_back({n_id_map[c], v});
            }
          }
          break;
        }
      }
    }

    for (int i = 1; i < idx.numel() + 1; i++) {
      out_rowptr_data[i] += out_rowptr_data[i - 1];
    }
    int64_t N = n_ids.size();
    auto out_n_id = torch::from_blob(n_ids.data(), {N}, idx.options()).clone();

    int64_t E = out_rowptr_data[idx.numel()];
    auto out_col = torch::empty(E, idx.options());
    auto out_col_data = out_col.data_ptr<int64_t>();
    std::optional<torch::Tensor> out_val;
    V *out_val_data = nullptr;
    if (has_val_) {
      if constexpr (std::is_same_v<V, float>) {
        out_val = torch::empty(E, idx.options().dtype(torch::kFloat32));
      } else {
        out_val = torch::empty(E, idx.options().dtype(torch::kInt32));
      }
      out_val_data = out_val->data_ptr<int64_t>();
    }

    i = 0;
    for (auto &col_vec : cols) {
      std::sort(col_vec.begin(), col_vec.end(),
                [](const auto &a, const auto &b) -> bool {
                  return std::get<0>(a) < std::get<0>(b);
                });
      for (const auto &value : col_vec) {
        out_col_data[i] = std::get<0>(value);
        if (has_val_) {
          (*out_val_data)[i] = std::get<1>(value);
        }
        i += 1;
      }
    }

    return {out_rowptr, out_col, out_n_id, out_val};
  }

private:
  std::size_t block_num_;
  std::size_t block_size_;
  std::size_t node_num_;
  std::vector<DynamicBlock<V>> blocks_;
  DynamicLogStore<V> logstore_;
  bool has_val_;

  std::size_t get_block(int64_t node) { return node / block_size_; }

  std::size_t get_offset(int64_t node) { return node % block_size_; }

  int64_t get_global(int64_t node, std::size_t block_id) {
    return block_id * block_size_ + node;
  }
};
