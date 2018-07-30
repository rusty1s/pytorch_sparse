#include <ATen/ATen.h>

#include <cusparse.h>

static cusparseHandle_t cusparse_handle = 0;

static void init_cusparse() {
  if (cusparse_handle == 0) {
    cusparseStatus_t status = cusparseCreate(&cusparse_handle);
  }
}

at::Tensor spspmm_cuda(at::Tensor matrix1, at::Tensor matrix2) {
  init_cusparse();

  auto nnz = matrix1._nnz();
  auto inDim = matrix1.size(1);

  auto row = matrix1._indices()[0].toType(at::kInt);
  auto row_ptrs = at::empty(row.type(), {inDim + 1});

  cusparseXcoo2csr(cusparse_handle, row.data<int>(), nnz, inDim,
                   row_ptrs.data<int>(), CUSPARSE_INDEX_BASE_ZERO);

  printf("%lli\n", nnz);
  printf("%lli\n", inDim);

  /* colbuf at::empty(nnz); */
  /* auto colPtrs = at::empty(inDim + 1, at::kInt); */

  /* auto row = matrix1._indices(); */
  /* for (int i = 0; i < 5; i++) { */
  /*   row_buf.data<int>()[i] = (int)row.data<int64_t>()[i]; */
  /* } */
  /* printf("%lli\n", row.numel()); */

  return matrix1;
}
/* #include <ATen/SparseTensorImpl.h> */

/* namespace at { */
/* namespace native { */
/* using SparseTensor = Tensor; */

/* namespace { */
/* at::SparseTensor spspmm_cuda(at::SparseTensor matrix1, */
/*                              at::SparseTensor matrix2) { */

/*   return matrix1; */
/* } */
/* } // namespace */
/* } // namespace native */
/* } // namespace at */

// defined in aten/src/THCUNN/SparseLinear.cu as

/* cusparseXcoo2csr(cusparse_handle, THCudaIntTensor_data(state, colbuf), nnz,
 */
/*                  inDim, THCudaIntTensor_data(state, colPtrs), */
/*                  CUSPARSE_INDEX_BASE_ONE); */
