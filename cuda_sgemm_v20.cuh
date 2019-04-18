#ifndef _CUDA_SGEMM_V20_CUH_
#define _CUDA_SGEMM_V20_CUH_

#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "matrix_helper.cuh"

template <int BLOCK_SIZE> __global__ void
kerSgemmV20(Matrix a, Matrix b, Matrix c)
{
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int tidx = threadIdx.x, tidy = threadIdx.y;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = devGetSubMatrix(c, bidx * BLOCK_SIZE * 2, bidy * BLOCK_SIZE * 2, BLOCK_SIZE * 2, BLOCK_SIZE * 2);

    // Each thread computes one element of Csub
    float Cvalue00 = 0.0f, Cvalue10 = 0.0f, Cvalue01 = 0.0f, Cvalue11 = 0.0f;

    for (int l = 0; l < a.cols; l += 4) {

        // Get sub-matrix Asub of A
        Matrix Asub = devGetSubMatrix(a, bidx * BLOCK_SIZE * 2, l, BLOCK_SIZE * 2, 4);

        // Get sub-matrix Bsub of B
        Matrix Bsub = devGetSubMatrix(b, l, bidy * BLOCK_SIZE * 2, 4, BLOCK_SIZE * 2);

        __shared__ float As[4][BLOCK_SIZE * 2];
        __shared__ float Bs[BLOCK_SIZE * 2][4];

        if (tidy < 4) {
            As[tidy][tidx] = devGetMatrixElement(Asub, tidx, tidy);
            As[tidy][tidx + BLOCK_SIZE] = devGetMatrixElement(Asub, tidx + BLOCK_SIZE, tidy);
        }

        if (tidx < 4) {
            Bs[tidy][tidx] = devGetMatrixElement(Bsub, tidx, tidy);
            Bs[tidy + BLOCK_SIZE][tidx] = devGetMatrixElement(Bsub, tidx, tidy + BLOCK_SIZE);
        }
        __syncthreads();

        float As00 = As[0][tidx * 2    ], As01 = As[1][tidx * 2    ], As02 = As[2][tidx * 2    ], As03 = As[3][tidx * 2    ];
        float As10 = As[0][tidx * 2 + 1], As11 = As[1][tidx * 2 + 1], As12 = As[2][tidx * 2 + 1], As13 = As[3][tidx * 2 + 1];
        float Bs00 = Bs[tidy * 2    ][0], Bs10 = Bs[tidy * 2    ][1], Bs20 = Bs[tidy * 2    ][2], Bs30 = Bs[tidy * 2    ][3];
        float Bs01 = Bs[tidy * 2 + 1][0], Bs11 = Bs[tidy * 2 + 1][1], Bs21 = Bs[tidy * 2 + 1][2], Bs31 = Bs[tidy * 2 + 1][3];

        Cvalue00 += As00 * Bs00;
        Cvalue00 += As01 * Bs10;
        Cvalue00 += As02 * Bs20;
        Cvalue00 += As03 * Bs30;

        Cvalue10 += As10 * Bs00;
        Cvalue10 += As11 * Bs10;
        Cvalue10 += As12 * Bs20;
        Cvalue10 += As13 * Bs30;

        Cvalue01 += As00 * Bs01;
        Cvalue01 += As01 * Bs11;
        Cvalue01 += As02 * Bs21;
        Cvalue01 += As03 * Bs31;

        Cvalue11 += As10 * Bs01;
        Cvalue11 += As11 * Bs11;
        Cvalue11 += As12 * Bs21;
        Cvalue11 += As13 * Bs31;
        
        __syncthreads();
    }

    devSetMatrixElement(Csub, tidx * 2,     tidy * 2,     Cvalue00);
    devSetMatrixElement(Csub, tidx * 2 + 1, tidy * 2,     Cvalue10);
    devSetMatrixElement(Csub, tidx * 2,     tidy * 2 + 1, Cvalue01);
    devSetMatrixElement(Csub, tidx * 2 + 1, tidy * 2 + 1, Cvalue11);
}

template <int BLOCK_SIZE> void 
hstCalByV20(Matrix a, Matrix b, Matrix c) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(c.rows / BLOCK_SIZE / 2, c.cols / BLOCK_SIZE / 2);
    kerWarmUp<<<1,1>>>();
    kerSgemmV20<BLOCK_SIZE><<<grid, block>>>(a, b, c);
}


#endif