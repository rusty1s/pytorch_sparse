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

std::tuple<at::Tensor, at::Tensor> spspmm_cuda(at::Tensor A, at::Tensor B) {
  init_cusparse();

  A = A.coalesce();
  B = B.coalesce();

  auto m = A.size(0);
  auto n = B.size(1);
  auto k = A.size(1);

  auto nnzA = A._nnz();
  auto nnzB = B._nnz();

  auto valueA = A._values();
  auto indexA = A._indices().toType(at::kInt);
  auto row_ptrA = at::empty(indexA.type(), {m + 1});
  cusparseXcoo2csr(cusparse_handle, indexA[0].data<int>(), nnzA, k,
                   row_ptrA.data<int>(), CUSPARSE_INDEX_BASE_ZERO);
  auto colA = indexA[1];
  cudaMemcpy(row_ptrA.data<int>() + m, &nnzA, sizeof(int),
             cudaMemcpyHostToDevice);

  auto valueB = B._values();
  auto indexB = B._indices().toType(at::kInt);
  auto row_ptrB = at::empty(indexB.type(), {k + 1});
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
  auto row_ptrC = at::empty(indexA.type(), {m + 1});
  cusparseXcsrgemmNnz(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnzA,
                      row_ptrA.data<int>(), colA.data<int>(), descr, nnzB,
                      row_ptrB.data<int>(), colB.data<int>(), descr,
                      row_ptrC.data<int>(), &nnzC);
  auto colC = at::empty(indexA.type(), {nnzC});
  auto valueC = at::empty(valueA.type(), {nnzC});

  CSRGEMM(valueC.type(), cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descr, nnzA,
          valueA.data<scalar_t>(), row_ptrA.data<int>(), colA.data<int>(),
          descr, nnzB, valueB.data<scalar_t>(), row_ptrB.data<int>(),
          colB.data<int>(), descr, valueC.data<scalar_t>(),
          row_ptrC.data<int>(), colC.data<int>());

  auto rowC = at::empty(indexA.type(), {nnzC});
  cusparseXcsr2coo(cusparse_handle, row_ptrC.data<int>(), nnzC, m,
                   rowC.data<int>(), CUSPARSE_INDEX_BASE_ZERO);

  auto indexC = at::stack({rowC, colC}, 0).toType(at::kLong);

  return std::make_tuple(indexC, valueC);
}
