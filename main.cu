// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "matrix.cuh"

#define M 4096
#define K 4096
#define N 4096

__global__ void warmup(){}

template <int blockDimX, int blockDimY> __global__ void
MatrixMulCUDA(float a, float b, float c, int m, int n, int k) {

}

void
calByRef(Matrix a, Matrix b, Matrix c);

int main()
{
	allocMatrixCPU(h_a, M, K);
	allocMatrixCPU(h_b, K, N);

	initMatrixa(h_a);
	initMatrixb(h_b);

	echoMatrix(h_a, 4);
	echoMatrix(h_b, 4);

	allocMatrixGPU(d_a, M, K);
	allocMatrixGPU(d_b, K, N);
	allocMatrixGPU(d_c, M, N);

    checkCudaErrors(cudaMemcpy(d_a.data, h_a.data, d_a.rows * d_a.cols * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b.data, h_b.data, d_b.rows * d_a.cols * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate with cublas
    allocMatrixCPU(h_ref, K, N);
    calByRef(d_a, d_b, d_c);
    checkCudaErrors(cudaMemcpy(h_ref.data, d_c.data, d_c.rows * d_c.cols * sizeof(float), cudaMemcpyDeviceToHost));
    echoMatrix(h_ref, 4);

/*


	//



    // Calculate with shared memory blocking
*/
}

void
calByRef(Matrix a, Matrix b, Matrix c) {
    float alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));
    warmup<<<1,1>>>();
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, a.rows, b.cols, a.cols, &alpha, a.data, a.stride, b.data, b.stride, &beta, c.data, c.stride));
    checkCudaErrors(cublasDestroy(handle));
}