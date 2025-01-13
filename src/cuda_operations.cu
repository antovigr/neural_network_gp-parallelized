#include <cuda.h>

__global__ void matrixMulKernel(const float* A, const float* B, float* C, const int A_rows, const int A_cols, const int B_cols) {
    // Compute the global thread index for both x and y dimensions
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Column index in C
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Row index in C

    // Check if the thread is within bounds
    if (idx < B_cols && idy < A_rows) {
        float value = 0.0f;
        // Perform the dot product for the row of A and column of B
        for (int k = 0; k < A_cols; ++k) {
            value += A[idy * A_cols + k] * B[k * B_cols + idx];
        }
        C[idy * B_cols + idx] = value;  // Store the result in C
    }
}

__global__ void addBiasToMatrixKernel(const float* matrix, const float* biases, float* result, int rows, int cols) {
    // Compute the global thread index for both x and y dimensions
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Row index in result matrix
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Column index in result matrix

    // Check if the thread is within bounds
    if (idx < rows && idy < cols) {
        // Add the bias to each element in the column
        result[idx * cols + idy] = matrix[idx * cols + idy] + biases[idy];
    }
}

__global__ void sigmoidKernel(const float* input, float* output, const int rows, const int cols) {
    // Compute the global thread index for both x and y dimensions
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within bounds
    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;  // Convert 2D index to 1D
        output[index] = 1.0f / (1.0f + expf(-input[index]));  // Sigmoid function
    }
}
