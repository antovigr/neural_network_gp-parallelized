__global__ void transposeKernel(float* d_input, float* d_output, int width, int height);

__global__ void gemmKernel(const float* A, const float* B, float* C, int M, int N, int K);

// CUDA kernel for adding a matrix (M, 1) to a matrix (M, N)
__global__ void addBiasKernel(const float* matrix, const float* bias, float* result, int M, int N);

__global__ void sigmoidKernel(const float* input, float* output, int size) ;

__global__ void subtractKernel(const float* A, const float* B, float* C, int rows, int cols);

__global__ void sigmoidDerivativeKernel(const float* input, float* output, int size);

__global__ void elementWiseMultiply(const float* A, const float* B, float* C, int size);

__global__ void matrixMultiplyTransposeKernel(float* A, float* B, float* C, int A_rows, int A_cols, int B_rows, int B_cols);

__global__ void elementWiseMultiplyScalarKernel(float* matrix, float scalar, float* result, int rows, int cols);

__global__ void matrixMeanRowKernel(float* matrix, float* result, int rows, int cols);

__global__ void elementWiseSubtractKernel(float* A, const float* B, float* C, int rows, int cols);