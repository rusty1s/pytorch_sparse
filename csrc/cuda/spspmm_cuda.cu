#include "spspmm_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <cusparse.h>

#include "utils.cuh"

#define AT_DISPATCH_CUSPARSE_TYPES(TYPE, ...)                                  \
  [&] {                                                                        \
    switch (TYPE) {                                                            \
    case torch::ScalarType::Float: {                                           \
      using scalar_t = float;                                                  \
      const auto &cusparsecsrgemm2_bufferSizeExt =                             \
          cusparseScsrgemm2_bufferSizeExt;                                     \
      const auto &cusparsecsrgemm2 = cusparseScsrgemm2;                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case torch::ScalarType::Double: {                                          \
      using scalar_t = double;                                                 \
      const auto &cusparsecsrgemm2_bufferSizeExt =                             \
          cusparseDcsrgemm2_bufferSizeExt;                                     \
      const auto &cusparsecsrgemm2 = cusparseDcsrgemm2;                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      AT_ERROR("Not implemented for '", toString(TYPE), "'");                  \
    }                                                                          \
  }()

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
spspmm_cuda(torch::Tensor rowptrA, torch::Tensor colA,
            torch::optional<torch::Tensor> optional_valueA,
            torch::Tensor rowptrB, torch::Tensor colB,
            torch::optional<torch::Tensor> optional_valueB, int64_t K,
            std::string reduce) {

  CHECK_CUDA(rowptrA);
  CHECK_CUDA(colA);
  if (optional_valueA.has_value())
    CHECK_CUDA(optional_valueA.value());
  CHECK_CUDA(rowptrB);
  CHECK_CUDA(colB);
  if (optional_valueB.has_value())
    CHECK_CUDA(optional_valueB.value());
  cudaSetDevice(rowptrA.get_device());

  CHECK_INPUT(rowptrA.dim() == 1);
  CHECK_INPUT(colA.dim() == 1);
  if (optional_valueA.has_value()) {
    CHECK_INPUT(optional_valueA.value().dim() == 1);
    CHECK_INPUT(optional_valueA.value().size(0) == colA.size(0));
  }
  CHECK_INPUT(rowptrB.dim() == 1);
  CHECK_INPUT(colB.dim() == 1);
  if (optional_valueB.has_value()) {
    CHECK_INPUT(optional_valueB.value().dim() == 1);
    CHECK_INPUT(optional_valueB.value().size(0) == colB.size(0));
  }

  if (!optional_valueA.has_value() && optional_valueB.has_value())
    optional_valueA =
        torch::ones(colA.numel(), optional_valueB.value().options());

  if (!optional_valueB.has_value() && optional_valueA.has_value())
    optional_valueB =
        torch::ones(colB.numel(), optional_valueA.value().options());

  auto scalar_type = torch::ScalarType::Float;
  if (optional_valueA.has_value())
    scalar_type = optional_valueA.value().scalar_type();

  auto handle = at::cuda::getCurrentCUDASparseHandle();

  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);

  rowptrA = rowptrA.toType(torch::kInt);
  colA = colA.toType(torch::kInt);
  rowptrB = rowptrB.toType(torch::kInt);
  colB = colB.toType(torch::kInt);

  int64_t M = rowptrA.numel() - 1, N = rowptrB.numel() - 1;
  auto rowptrA_data = rowptrA.data_ptr<int>();
  auto colA_data = colA.data_ptr<int>();
  auto rowptrB_data = rowptrB.data_ptr<int>();
  auto colB_data = colB.data_ptr<int>();

  torch::Tensor rowptrC, colC;
  torch::optional<torch::Tensor> optional_valueC = torch::nullopt;

  int nnzC;
  int *nnzTotalDevHostPtr = &nnzC;

  // Step 1: Create an opaque structure.
  csrgemm2Info_t info = NULL;
  cusparseCreateCsrgemm2Info(&info);

  // Step 2: Allocate buffer for `csrgemm2Nnz` and `csrgemm2`.
  size_t bufferSize;
  AT_DISPATCH_CUSPARSE_TYPES(scalar_type, [&] {
    scalar_t alpha = (scalar_t)1.0;
    cusparsecsrgemm2_bufferSizeExt(handle, M, N, K, &alpha, descr, colA.numel(),
                                   rowptrA_data, colA_data, descr, colB.numel(),
                                   rowptrB_data, colB_data, NULL, descr, 0,
                                   NULL, NULL, info, &bufferSize);

    void *buffer = NULL;
    cudaMalloc(&buffer, bufferSize);

    // Step 3: Compute CSR row pointer.
    rowptrC = torch::empty(M + 1, rowptrA.options());
    auto rowptrC_data = rowptrC.data_ptr<int>();
    cusparseXcsrgemm2Nnz(handle, M, N, K, descr, colA.numel(), rowptrA_data,
                         colA_data, descr, colB.numel(), rowptrB_data,
                         colB_data, descr, 0, NULL, NULL, descr, rowptrC_data,
                         nnzTotalDevHostPtr, info, buffer);

    // Step 4: Compute CSR entries.
    colC = torch::empty(nnzC, rowptrC.options());
    auto colC_data = colC.data_ptr<int>();

    if (optional_valueA.has_value())
      optional_valueC = torch::empty(nnzC, optional_valueA.value().options());

    scalar_t *valA_data = NULL, *valB_data = NULL, *valC_data = NULL;
    if (optional_valueA.has_value()) {
      valA_data = optional_valueA.value().data_ptr<scalar_t>();
      valB_data = optional_valueB.value().data_ptr<scalar_t>();
      valC_data = optional_valueC.value().data_ptr<scalar_t>();
    }

    cusparsecsrgemm2(handle, M, N, K, &alpha, descr, colA.numel(), valA_data,
                     rowptrA_data, colA_data, descr, colB.numel(), valB_data,
                     rowptrB_data, colB_data, NULL, descr, 0, NULL, NULL, NULL,
                     descr, valC_data, rowptrC_data, colC_data, info, buffer);

    cudaFree(buffer);
  });

  // Step 5: Destroy the opaque structure.
  cusparseDestroyCsrgemm2Info(info);

  rowptrC = rowptrC.toType(torch::kLong);
  colC = colC.toType(torch::kLong);

  return std::make_tuple(rowptrC, colC, optional_valueC);
}
