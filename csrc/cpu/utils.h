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
