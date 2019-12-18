#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

at::Tensor spmm_cuda(at::Tensor rowptr, at::Tensor col,
                     at::optional<at::Tensor> val, at::Tensor mat,
                     std::string reduce);

std::tuple<at::Tensor, at::Tensor>
spmm_arg_cuda(at::Tensor rowptr, at::Tensor col, at::optional<at::Tensor> val,
              at::Tensor mat, std::string reduce);

at::Tensor spmm(at::Tensor rowptr, at::Tensor col, at::optional<at::Tensor> val,
                at::Tensor mat, std::string reduce) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  if (val.has_value())
    CHECK_CUDA(val.value());
  CHECK_CUDA(mat);
  return spmm_cuda(rowptr, col, val, mat, reduce);
}

std::tuple<at::Tensor, at::Tensor> spmm_arg(at::Tensor rowptr, at::Tensor col,
                                            at::optional<at::Tensor> val,
                                            at::Tensor mat,
                                            std::string reduce) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  if (val.has_value())
    CHECK_CUDA(val.value());
  CHECK_CUDA(mat);
  return spmm_arg_cuda(rowptr, col, val, mat, reduce);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm", &spmm, "Sparse Matrix Multiplication (CUDA)");
  m.def("spmm_arg", &spmm_arg, "Sparse Matrix Multiplication With Arg (CUDA)");
}
