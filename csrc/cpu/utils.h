#pragma once

#include <torch/extension.h>

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
torch::Tensor from_vector(const std::vector<scalar_t> &vec,
                          bool inplace = false) {
  const auto size = (int64_t)vec.size();
  const auto out = torch::from_blob((scalar_t *)vec.data(), {size},
                                    c10::CppTypeToScalarType<scalar_t>::value);
  return inplace ? out : out.clone();
}

torch::Tensor choice(int64_t population, int64_t num_samples,
                     bool replace = false,
                     torch::optional<torch::Tensor> weight = torch::nullopt) {

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
    std::unordered_set<int64_t> values;
    for (int64_t i = population - num_samples; i < population; i++) {
      if (!values.insert(rand() % i).second)
        values.insert(i);
    }
    const auto out = torch::empty(num_samples, at::kLong);
    auto *out_data = out.data_ptr<int64_t>();
    int64_t i = 0;
    for (const auto &value : values) {
      out2_data[i] = value;
      i++;
    }
    return out;
  }
}
