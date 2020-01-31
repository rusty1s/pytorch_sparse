#pragma once

#include <torch/extension.h>

torch::Tensor ind2ptr_cuda(torch::Tensor ind, int64_t M);
torch::Tensor ptr2ind_cuda(torch::Tensor ptr, int64_t E);
