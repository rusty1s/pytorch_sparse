#pragma once

static inline __device__ void atomAdd(float *address, float val) {
  atomicAdd(address, val);
}

static inline __device__ void atomAdd(double *address, double val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
#else
  atomicAdd(address, val);
#endif
}
