#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cusparse.h>

#include "compat.cuh"

#define AT_DISPATCH_CUSPARSE_CSR_GEMM2_BUFFER_SIZE_EXT_TYPES(TYPE, ...)        \
  [&] {                                                                        \
    switch (TYPE) {                                                            \
    case torch::ScalarType::Float: {                                           \
      using scalar_t = float;                                                  \
      const auto &cusparsecsrgemm2_bufferSizeExt =                             \
          cusparseScsrgemm2_bufferSizeExt;                                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case torch::ScalarType::Double: {                                          \
      using scalar_t = double;                                                 \
      const auto &cusparsecsrgemm2_bufferSizeExt =                             \
          cusparseDcsrgemm2_bufferSizeExt;                                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      AT_ERROR("Not implemented for '", toString(TYPE), "'");                  \
    }                                                                          \
  }()

#define AT_DISPATCH_CUSPARSE_CSR_GEMM2_TYPES(TYPE, ...)                        \
  [&] {                                                                        \
    switch (TYPE) {                                                            \
    case torch::ScalarType::Float: {                                           \
      using scalar_t = float;                                                  \
      const auto &cusparsecsrgemm2 = cusparseScsrgemm2;                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case torch::ScalarType::Double: {                                          \
      using scalar_t = double;                                                 \
      const auto &cusparsecsrgemm2 = cusparseDcsrgemm2;                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      AT_ERROR("Not implemented for '", toString(TYPE), "'");                  \
    }                                                                          \
  }()

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
spspmm_cuda(torch::Tensor rowptrA, torch::Tensor colA,
            torch::optional<torch::Tensor> valueA, torch::Tensor rowptrB,
            torch::Tensor colB, torch::optional<torch::Tensor> valueB,
            int64_t M, int64_t N, int64_t K) {
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  auto handle = at::cuda::getCurrentCUDASparseHandle();

  rowptrA = rowptrA.toType(torch::kInt), colA = colA.toType(torch::kInt);
  rowptrB = rowptrB.toType(torch::kInt), colB = colB.toType(torch::kInt);

  auto rowptrA_data = rowptrA.DATA_PTR<int>(), colA_data = colA.DATA_PTR<int>();
  auto rowptrB_data = rowptrB.DATA_PTR<int>(), colB_data = colB.DATA_PTR<int>();

  csrgemm2Info_t info = NULL;
  cusparseCreateCsrgemm2Info(&info);

  auto scalar_type = torch::ScalarType::Float;
  if (valueA.has_value())
    scalar_type = valueA.value().scalar_type();
  if (valueB.has_value())
    scalar_type = valueB.value().scalar_type();

  size_t bufferSize;
  AT_DISPATCH_CUSPARSE_CSR_GEMM2_BUFFER_SIZE_EXT_TYPES(scalar_type, [&] {
    scalar_t alpha = (scalar_t)1;
    cusparsecsrgemm2_bufferSizeExt(handle, M, N, K, &alpha, descr, colA.numel(),
                                   rowptrA_data, colA_data, descr, colB.numel(),
                                   rowptrB_data, colB_data, NULL, descr, 0,
                                   NULL, NULL, info, &bufferSize);
  });

  void *buffer = NULL;
  cudaMalloc(&buffer, bufferSize);

  int nnzC;
  auto rowptrC = torch::empty(M + 1, rowptrA.options());
  auto rowptrC_data = rowptrC.DATA_PTR<int>();
  cusparseXcsrgemm2Nnz(handle, M, N, K, descr, colA.numel(), rowptrA_data,
                       colA_data, descr, colB.numel(), rowptrB_data, colB_data,
                       descr, 0, NULL, NULL, descr, rowptrC_data, &nnzC, info,
                       buffer);

  auto colC = torch::empty(nnzC, colA.options());
  auto colC_data = colC.DATA_PTR<int>();

  if (!valueA.has_value() && valueB.has_value())
    valueA = torch::ones_like(valueB.value());

  if (!valueB.has_value() && valueA.has_value())
    valueB = torch::ones_like(valueA.value());

  torch::optional<torch::Tensor> valueC = torch::nullopt;
  if (valueA.has_value())
    valueC = torch::empty(nnzC, valueA.value().options());

  AT_DISPATCH_CUSPARSE_CSR_GEMM2_TYPES(scalar_type, [&] {
    scalar_t alpha = (scalar_t)1;

    scalar_t *valueA_data = NULL;
    if (valueA.has_value())
      valueA_data = valueA.value().DATA_PTR<scalar_t>();

    scalar_t *valueB_data = NULL;
    if (valueB.has_value())
      valueB_data = valueB.value().DATA_PTR<scalar_t>();

    scalar_t *valueC_data = NULL;
    if (valueC.has_value())
      valueC_data = valueC.value().DATA_PTR<scalar_t>();

    cusparsecsrgemm2(handle, M, N, K, &alpha, descr, colA.numel(), valueA_data,
                     rowptrA_data, colA_data, descr, colB.numel(), valueB_data,
                     rowptrB_data, colB_data, NULL, descr, 0, NULL, NULL, NULL,
                     descr, valueC_data, rowptrC_data, colC_data, info, buffer);
  });

  rowptrC = rowptrC.toType(torch::kLong);
  colC = colC.toType(torch::kLong);

  return std::make_tuple(rowptrC, colC, valueC);
}
