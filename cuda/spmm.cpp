#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

std::tuple<at::Tensor, at::optional<at::Tensor>>
spmm_cuda(at::Tensor rowptr, at::Tensor col, at::optional<at::Tensor> value_opt,
          at::Tensor mat, std::string reduce);

at::Tensor spmm_val_bw_cuda(at::Tensor index, at::Tensor rowptr, at::Tensor mat,
                            at::Tensor grad, std::string reduce);

std::tuple<at::Tensor, at::optional<at::Tensor>>
spmm(at::Tensor rowptr, at::Tensor col, at::optional<at::Tensor> value_opt,
     at::Tensor mat, std::string reduce) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  if (value_opt.has_value())
    CHECK_CUDA(value_opt.value());
  CHECK_CUDA(mat);
  return spmm_cuda(rowptr, col, value_opt, mat, reduce);
}

at::Tensor spmm_val_bw(at::Tensor index, at::Tensor rowptr, at::Tensor mat,
                       at::Tensor grad, std::string reduce) {
  CHECK_CUDA(index);
  CHECK_CUDA(rowptr);
  CHECK_CUDA(mat);
  CHECK_CUDA(grad);
  return spmm_val_bw_cuda(index, rowptr, mat, grad, reduce);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm", &spmm, "Sparse Matrix Multiplication (CUDA)");
  m.def("spmm_val_bw", &spmm_val_bw,
        "Sparse-Dense Matrix Multiplication Value Backward (CPU)");
}
