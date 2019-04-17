// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "cuda_sgemm_cublas.cuh"
#include "cuda_sgemm_v10.cuh"

#define M 4096
#define K 4096
#define N 4096


int main()
{
	hstAllocMatrix(h_a, M, K);
	hstAllocMatrix(h_b, K, N);

	hstInitMatrix(h_a);
	hstInitMatrix(h_b);

	hstEchoMatrix(h_a, 4);
	hstEchoMatrix(h_b, 4);

	kerAllocMatrix(d_a, M, K);
	kerAllocMatrix(d_b, K, N);
	kerAllocMatrix(d_c, M, N);

    checkCudaErrors(cudaMemcpy(d_a.data, h_a.data, M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b.data, h_b.data, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate with cublas
    hstAllocMatrix(h_ref, M, N);
    hstCalByRef(d_a, d_b, d_c);
    checkCudaErrors(cudaMemcpy(h_ref.data, d_c.data, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    hstEchoMatrix(h_ref, 4);

    // Calculate with cuda_sgemm_v1
    hstAllocMatrix(h_v10, M, N);
    hstCalByV10<32>(d_a, d_b, d_c);
    checkCudaErrors(cudaMemcpy(h_v10.data, d_c.data, d_c.rows * d_c.cols * sizeof(float), cudaMemcpyDeviceToHost));
    hstEchoMatrix(h_v10, 4);
/*


	//



    // Calculate with shared memory blocking
*/
}



