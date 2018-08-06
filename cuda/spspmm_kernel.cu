#include <ATen/ATen.h>

#include <cusparse.h>

#define CSRGEMM(TYPE, ...)                                                     \
  [&] {                                                                        \
    const at::Type &the_type = TYPE;                                           \
    switch (the_type.scalarType()) {                                           \
    case at::ScalarType::Float: {                                              \
      using scalar_t = float;                                                  \
      return cusparseScsrgemm(__VA_ARGS__);                                    \
    }                                                                          \
    case at::ScalarType::Double: {                                             \
      using scalar_t = double;                                                 \
      return cusparseDcsrgemm(__VA_ARGS__);                                    \
    }                                                                          \
    default:                                                                   \
      AT_ERROR("Not implemented for '%s'", the_type.toString());               \
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
            at::Tensor valueB, int m, int k, int n) {
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
  auto row_ptrA = at::empty(m + 1, indexA.type());
  cusparseXcoo2csr(cusparse_handle, indexA[0].data<int>(), nnzA, k,
                   row_ptrA.data<int>(), CUSPARSE_INDEX_BASE_ZERO);
  auto colA = indexA[1];
  cudaMemcpy(row_ptrA.data<int>() + m, &nnzA, sizeof(int),
             cudaMemcpyHostToDevice);

  // Convert B to CSR format.
  auto row_ptrB = at::empty(k + 1, indexB.type());
  cusparseXcoo2csr(cusparse_handle, indexB[0].data<int>(), nnzB, k,
                   row_ptrB.data<int>(), CUSPARSE_INDEX_BASE_ZERO);
  auto colB = indexB[1];
  cudaMemcpy(row_ptrB.data<int>() + k, &nnzB, sizeof(int),
             cudaMemcpyHostToDevice);

  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  int nnzC;
  auto row_ptrC = at::empty(m + 1, indexB.type());
  cusparseXcsrgemmNnz(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnzA,
                      row_ptrA.data<int>(), colA.data<int>(), descr, nnzB,
                      row_ptrB.data<int>(), colB.data<int>(), descr,
                      row_ptrC.data<int>(), &nnzC);
  auto colC = at::empty(nnzC, indexA.type());
  auto valueC = at::empty(nnzC, valueA.type());

  CSRGEMM(valueC.type(), cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnzA,
          valueA.data<scalar_t>(), row_ptrA.data<int>(), colA.data<int>(),
          descr, nnzB, valueB.data<scalar_t>(), row_ptrB.data<int>(),
          colB.data<int>(), descr, valueC.data<scalar_t>(),
          row_ptrC.data<int>(), colC.data<int>());

  auto rowC = at::empty(nnzC, indexA.type());
  cusparseXcsr2coo(cusparse_handle, row_ptrC.data<int>(), nnzC, m,
                   rowC.data<int>(), CUSPARSE_INDEX_BASE_ZERO);

  auto indexC = at::stack({rowC, colC}, 0).toType(at::kLong);

  return std::make_tuple(indexC, valueC);
}
