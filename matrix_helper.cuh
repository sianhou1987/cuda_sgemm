#ifndef _MATRIX_HELPER_CUH_
#define _MATRIX_HELPER_CUH_

typedef struct {
	int rows;
	int cols;
	int stride;
	float *data;
} Matrix;

#define hstAllocMatrix(a, m, n) Matrix a; a.rows = m; a. cols = n; a.stride = m; a.data = (float *)malloc(m * n * sizeof(float));
#define kerAllocMatrix(a, m, n) Matrix a; a.rows = m; a. cols = n; a.stride = m; cudaMalloc((void **)&a.data, m * n * sizeof(float));

float hstGetMatrixElement(Matrix a, int row, int col)
{
	return a.data[col * a.stride + row];
}

void hstSetMatrixElement(Matrix a, int row, int col, float value)
{
	a.data[col * a.stride + row] = value;
}

void hstInitMatrix(Matrix a)
{
	for (int row = 0; row < a.rows; ++row)
		for (int col = 0; col < a.cols; ++col)
			hstSetMatrixElement(a, row, col, row - col);
}

void hstSetMatrix(Matrix a, float value)
{
	for (int row = 0; row < a.rows; ++row)
		for (int col = 0; col < a.cols; ++col)
			hstSetMatrixElement(a, row, col, value);
}

void hstEchoMatrix(Matrix a, int n)
{
	for (int row = 0; row < n; ++row) {
		for (int col = 0; col < n; ++col) {
			printf("%f ", hstGetMatrixElement(a, row, col));
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void 
kerWarmUp(){}

// Get a matrix element
__device__ float
devGetMatrixElement(const Matrix a, int row, int col)
{
    return a.data[col * a.stride + row];
}

// Set a matrix element
__device__ void
devSetMatrixElement(Matrix a, int row, int col, float value)
{
    a.data[col * a.stride + row] = value;
}

// Set a sub matrix element
__device__ Matrix
devGetSubMatrix(Matrix a, int row, int col, int m, int n)
{
    Matrix sub;
    sub.rows = m;
    sub.cols = n;
    sub.stride = a.stride;
    sub.data = &a.data[col * a.stride + row];
    return sub;
}


#endif