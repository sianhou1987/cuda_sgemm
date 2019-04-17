#ifndef _CUDA_SGEMM_V10_CUH_
#define _CUDA_SGEMM_V10_CUH_

#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "matrix_helper.cuh"

/**
  * Calculate sgemm with shared memory blocking algorithm, copied from NVIDIA's PROGRAMMING GUIDE
  **/

template <int BLOCK_SIZE> __global__ void
kerSgemmV10(Matrix a, Matrix b, Matrix c)
{
    int dimx = blockDim.x, dimy = blockDim.y;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int tidx = threadIdx.x, tidy = threadIdx.y;
    
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = devGetSubMatrix(c, bidx * dimx, bidy * dimy, BLOCK_SIZE, BLOCK_SIZE);

    // Each thread computes one element of Csub
    float Cvalue = 0.0f;

    for (int l = 0; l < a.cols; l += BLOCK_SIZE) {

        // Get sub-matrix Asub of A
        Matrix Asub = devGetSubMatrix(a, bidx * dimx, l, BLOCK_SIZE, BLOCK_SIZE);

        // Get sub-matrix Bsub of B
        Matrix Bsub = devGetSubMatrix(b, l, bidy * dimy, BLOCK_SIZE, BLOCK_SIZE);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];     
        
        As[tidy][tidx] = devGetMatrixElement(Asub, tidx, tidy);
        Bs[tidy][tidx] = devGetMatrixElement(Bsub, tidx, tidy);
        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[e][tidx] * Bs[tidy][e];
        __syncthreads();
        
    }

    devSetMatrixElement(Csub, tidx, tidy, Cvalue);
}

template <int BLOCK_SIZE> void 
hstCalByV10(Matrix a, Matrix b, Matrix c) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(c.rows / BLOCK_SIZE, c.cols / BLOCK_SIZE);
    kerSgemmV10<BLOCK_SIZE><<<grid, block>>>(a, b, c);
}

#endif