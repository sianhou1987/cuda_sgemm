#ifndef _CUDA_SGEMM_CUBLAS_CUH_
#define _CUDA_SGEMM_CUBLAS_CUH_

#include "matrix_helper.cuh"

void
hstCalByRef(Matrix a, Matrix b, Matrix c) {
    float alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));
    kerWarmUp<<<1,1>>>();
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, a.rows, b.cols, a.cols, &alpha, a.data, a.stride, b.data, b.stride, &beta, c.data, c.stride));
    checkCudaErrors(cublasDestroy(handle));
}

#endif