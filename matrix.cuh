typedef struct {
	int rows;
	int cols;
	int stride;
	float *data;
} Matrix;

#define allocMatrixCPU(a, m, n) Matrix a; a.rows = m; a. cols = n; a.stride = m; a.data = (float *)malloc(m * n * sizeof(float));
#define allocMatrixGPU(a, m, n) Matrix a; a.rows = m; a. cols = n; a.stride = m; cudaMalloc((void **)&a.data, m * n * sizeof(float));

void initMatrixa(Matrix a)
{
	for (int i = 0; i < a.rows; i++)
		for (int j = 0; j < a.cols; j++)
			a.data[j * a.stride + i] = j + i;
}

void initMatrixb(Matrix a)
{
	for (int i = 0; i < a.rows; i++)
		for (int j = 0; j < a.cols; j++)
			a.data[j * a.stride + i] = j - i;
}

void echoMatrix(Matrix a, int n)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f ", a.data[j * a.stride + i]);
		}
		printf("\n");
	}
	printf("\n");
}