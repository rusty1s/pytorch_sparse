#pragma once

#include <torch/extension.h>

torch::Tensor non_diag_mask_cpu(torch::Tensor row, torch::Tensor col, int64_t M,
                                int64_t N, int64_t k);
