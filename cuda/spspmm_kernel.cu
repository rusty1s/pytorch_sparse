#include <ATen/ATen.h>
#include <cusparse.h>

#include "compat.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

#define CSRGEMM(TYPE, ...)                                                     \
  [&] {                                                                        \
    const auto &the_type = TYPE;                                               \
    (void)the_type;                                                            \
    at::ScalarType _st = ::detail::scalar_type(TYPE);                          \
    switch (_st) {                                                             \
    case at::ScalarType::Float: {                                              \
      using scalar_t = float;                                                  \
      return cusparseScsrgemm(__VA_ARGS__);                                    \
    }                                                                          \
    case at::ScalarType::Double: {                                             \
      using scalar_t = double;                                                 \
      return cusparseDcsrgemm(__VA_ARGS__);                                    \
    }                                                                          \
    default:                                                                   \
      AT_ERROR("Not implemented for '", toString(_st), "'");                   \
    }                                                                          \
  }()

static cusparseHandle_t cusparse_handle = 0;

static void init_cusparse() {
  if (cusparse_handle == 0) {
    cusparseStatus_t status = cusparseCreate(&cusparse_handle);
  }
}

std::tuple<at::Tensor, at::Tensor>
spspmm_cuda(at::Tensor indexA, at::Tensor valueA, at::Tensor indexB,
            at::Tensor valueB, size_t m, size_t k, size_t n) {
  cudaSetDevice(indexA.get_device());
  init_cusparse();

  indexA = indexA.contiguous();
  valueA = valueA.contiguous();
  indexB = indexB.contiguous();
  valueB = valueB.contiguous();

  auto nnzA = valueA.size(0);
  auto nnzB = valueB.size(0);

  indexA = indexA.toType(at::kInt);
  indexB = indexB.toType(at::kInt);

  // Convert A to CSR format.
  auto row_ptrA = at::empty(m + 1, indexA.options());
  cusparseXcoo2csr(cusparse_handle, indexA[0].DATA_PTR<int>(), nnzA, k,
                   row_ptrA.DATA_PTR<int>(), CUSPARSE_INDEX_BASE_ZERO);
  auto colA = indexA[1];
  cudaMemcpy(row_ptrA.DATA_PTR<int>() + m, &nnzA, sizeof(int),
             cudaMemcpyHostToDevice);

  // Convert B to CSR format.
  auto row_ptrB = at::empty(k + 1, indexB.options());
  cusparseXcoo2csr(cusparse_handle, indexB[0].DATA_PTR<int>(), nnzB, k,
                   row_ptrB.DATA_PTR<int>(), CUSPARSE_INDEX_BASE_ZERO);
  auto colB = indexB[1];
  cudaMemcpy(row_ptrB.DATA_PTR<int>() + k, &nnzB, sizeof(int),
             cudaMemcpyHostToDevice);

  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  int nnzC;
  auto row_ptrC = at::empty(m + 1, indexB.options());
  cusparseXcsrgemmNnz(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnzA,
                      row_ptrA.DATA_PTR<int>(), colA.DATA_PTR<int>(), descr,
                      nnzB, row_ptrB.DATA_PTR<int>(), colB.DATA_PTR<int>(),
                      descr, row_ptrC.DATA_PTR<int>(), &nnzC);
  auto colC = at::empty(nnzC, indexA.options());
  auto valueC = at::empty(nnzC, valueA.options());

  CSRGEMM(valueC.scalar_type(), cusparse_handle,
          CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m,
          n, k, descr, nnzA, valueA.DATA_PTR<scalar_t>(),
          row_ptrA.DATA_PTR<int>(), colA.DATA_PTR<int>(), descr, nnzB,
          valueB.DATA_PTR<scalar_t>(), row_ptrB.DATA_PTR<int>(),
          colB.DATA_PTR<int>(), descr, valueC.DATA_PTR<scalar_t>(),
          row_ptrC.DATA_PTR<int>(), colC.DATA_PTR<int>());

  auto rowC = at::empty(nnzC, indexA.options());
  cusparseXcsr2coo(cusparse_handle, row_ptrC.DATA_PTR<int>(), nnzC, m,
                   rowC.DATA_PTR<int>(), CUSPARSE_INDEX_BASE_ZERO);

  auto indexC = at::stack({rowC, colC}, 0).toType(at::kLong);

  return std::make_tuple(indexC, valueC);
}

at::Tensor degree(at::Tensor row, int64_t num_nodes) {
  auto zero = at::zeros(num_nodes, row.options());
  auto one = at::ones(row.size(0), row.options());
  return zero.scatter_add_(0, row, one);
}

std::tuple<at::Tensor, at::Tensor> to_csr(at::Tensor row, at::Tensor col,
                                          int64_t num_nodes) {
  // Assert already coalesced input.
  row = degree(row, num_nodes).cumsum(0);
  row = at::cat({at::zeros(1, row.options()), row}, 0); // Prepend zero.
  return std::make_tuple(row, col);
}

template <typename scalar_t>
__global__ void spspmm_bw_kernel(
    const int64_t *__restrict__ index, scalar_t *__restrict__ value,
    const int64_t *__restrict__ rowA, const int64_t *__restrict__ colA,
    const scalar_t *__restrict__ valueA, const int64_t *__restrict__ rowB,
    const int64_t *__restrict__ colB, const scalar_t *__restrict__ valueB,
    const size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t e = idx; e < numel; e += stride) {
    int64_t i = index[e], j = index[numel + e];

    for (ptrdiff_t dA = rowA[i]; dA < rowA[i + 1]; dA++) {
      int64_t cA = colA[dA];

      for (ptrdiff_t dB = rowB[j]; dB < rowB[j + 1]; dB++) {
        int64_t cB = colB[dB];

        if (cA == cB) {
          value[e] += valueA[dA] * valueB[dB];
        }

        if (cB >= cA) {
          break;
        }
      }
    }
  }
}

at::Tensor spspmm_bw_cuda(at::Tensor index, at::Tensor indexA,
                          at::Tensor valueA, at::Tensor indexB,
                          at::Tensor valueB, size_t rowA_max, size_t rowB_max) {
  cudaSetDevice(index.get_device());
  auto value = at::zeros(index.size(1), valueA.options());

  at::Tensor rowA, colA;
  std::tie(rowA, colA) = to_csr(indexA[0], indexA[1], rowA_max);

  at::Tensor rowB, colB;
  std::tie(rowB, colB) = to_csr(indexB[0], indexB[1], rowB_max);

  AT_DISPATCH_FLOATING_TYPES(valueA.scalar_type(), "spspmm_bw", [&] {
    spspmm_bw_kernel<scalar_t><<<BLOCKS(value.numel()), THREADS>>>(
        index.DATA_PTR<int64_t>(), value.DATA_PTR<scalar_t>(),
        rowA.DATA_PTR<int64_t>(), colA.DATA_PTR<int64_t>(),
        valueA.DATA_PTR<scalar_t>(), rowB.DATA_PTR<int64_t>(),
        colB.DATA_PTR<int64_t>(), valueB.DATA_PTR<scalar_t>(), value.numel());
  });

  return value;
}
