#pragma once

#include "../extensions.h"

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

#define AT_DISPATCH_HAS_VALUE(optional_value, ...)                             \
  [&] {                                                                        \
    if (optional_value.has_value()) {                                          \
      const bool HAS_VALUE = true;                                             \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      const bool HAS_VALUE = false;                                            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

template <typename scalar_t>
inline torch::Tensor from_vector(const std::vector<scalar_t> &vec,
                                 bool inplace = false) {
  const auto size = (int64_t)vec.size();
  const auto out = torch::from_blob((scalar_t *)vec.data(), {size},
                                    c10::CppTypeToScalarType<scalar_t>::value);
  return inplace ? out : out.clone();
}

template <typename key_t, typename scalar_t>
inline c10::Dict<key_t, torch::Tensor>
from_vector(const std::unordered_map<key_t, std::vector<scalar_t>> &vec_dict,
            bool inplace = false) {
  c10::Dict<key_t, torch::Tensor> out_dict;
  for (const auto &kv : vec_dict)
    out_dict.insert(kv.first, from_vector<scalar_t>(kv.second, inplace));
  return out_dict;
}

inline torch::Tensor
choice(int64_t population, int64_t num_samples, bool replace = false,
       torch::optional<torch::Tensor> weight = torch::nullopt) {

  if (population == 0 || num_samples == 0)
    return torch::empty({0}, at::kLong);

  if (!replace && num_samples >= population)
    return torch::arange(population, at::kLong);

  if (weight.has_value())
    return torch::multinomial(weight.value(), num_samples, replace);

  if (replace) {
    const auto out = torch::empty(num_samples, at::kLong);
    auto *out_data = out.data_ptr<int64_t>();
    for (int64_t i = 0; i < num_samples; i++) {
      out_data[i] = rand() % population;
    }
    return out;

  } else {
    // Sample without replacement via Robert Floyd algorithm:
    // https://www.nowherenearithaca.com/2013/05/
    // robert-floyds-tiny-and-beautiful.html
    const auto out = torch::empty(num_samples, at::kLong);
    auto *out_data = out.data_ptr<int64_t>();
    std::unordered_set<int64_t> samples;
    for (int64_t i = population - num_samples; i < population; i++) {
      int64_t sample = rand() % i;
      if (!samples.insert(sample).second) {
        sample = i;
        samples.insert(sample);
      }
      out_data[i - population + num_samples] = sample;
    }
    return out;
  }
}

template <bool replace>
inline void
uniform_choice(const int64_t population, const int64_t num_samples,
               const int64_t *idx_data, std::vector<int64_t> *samples,
               std::unordered_map<int64_t, int64_t> *to_local_node) {

  if (population == 0 || num_samples == 0)
    return;

  if (replace) {
    for (int64_t i = 0; i < num_samples; i++) {
      const int64_t &v = idx_data[rand() % population];
      if (to_local_node->insert({v, samples->size()}).second)
        samples->push_back(v);
    }
  } else if (num_samples >= population) {
    for (int64_t i = 0; i < population; i++) {
      const int64_t &v = idx_data[i];
      if (to_local_node->insert({v, samples->size()}).second)
        samples->push_back(v);
    }
  } else {
    std::unordered_set<int64_t> indices;
    for (int64_t i = population - num_samples; i < population; i++) {
      int64_t j = rand() % i;
      if (!indices.insert(j).second) {
        j = i;
        indices.insert(j);
      }
      const int64_t &v = idx_data[j];
      if (to_local_node->insert({v, samples->size()}).second)
        samples->push_back(v);
    }
  }
}
