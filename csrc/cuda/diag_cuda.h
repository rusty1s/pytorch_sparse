#pragma once

#include "../extensions.h"

torch::Tensor non_diag_mask_cuda(torch::Tensor row, torch::Tensor col,
                                 int64_t M, int64_t N, int64_t k);
