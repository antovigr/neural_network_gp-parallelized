#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

__global__ void transposeKernel(float* d_input, float* d_output, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        int input_idx = j * width + i;  // 1D index of element in the input matrix
        int output_idx = i * height + j;  // 1D index of element in the output matrix
        d_output[output_idx] = d_input[input_idx];
    }
}

__global__ void gemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // M: Rows in A, C
    // N: Columns in B, C
    // K: Columns in A and Rows in B
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index in C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index in C

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matrixMultiplyTransposeKernel(float* A, float* B, float* C, int A_rows, int A_cols, int B_rows, int B_cols) {
    // Calculate the row and column indices for the thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        // Multiply matrix A with transpose of matrix B
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[col * B_rows + k]; // B[col, k] because B is transposed
        }
        C[row * B_cols + col] = sum;
    }
}


// CUDA kernel for adding a matrix (M, 1) to a matrix (M, N)
__global__ void addBiasKernel(const float* matrix, const float* bias, float* result, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

    if (row < M && col < N) {
        result[row * N + col] = matrix[row * N + col] + bias[row];
    }
}

__global__ void sigmoidKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Compute global thread index
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void subtractKernel(const float* A, const float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Compute global thread index

    // Ensure we are within bounds
    if (idx < rows * cols) {
        C[idx] = A[idx] - B[idx];  // Subtract corresponding elements
    }
}

__global__ void sigmoidDerivativeKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double value = input[idx];
        double sigmoid_val = 1.0 / (1.0 + exp(-value));  // Sigmoid function
        output[idx] = sigmoid_val * (1.0 - sigmoid_val);  // Sigmoid derivative
    }
}

__global__ void elementWiseMultiply(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];  // Element-wise multiplication
    }
}

__global__ void elementWiseMultiplyScalarKernel(float* matrix, float scalar, float* result, int rows, int cols) {
    // Calculate the row and column indices for the thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (row < rows && col < cols) {
        // Perform element-wise multiplication by scalar
        result[row * cols + col] = matrix[row * cols + col] * scalar;
    }
}

__global__ void matrixMeanRowKernel(float* matrix, float* result, int rows, int cols) {
    // Calculate the column index for the thread (result will have one value per column)
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within bounds
    if (col < cols) {
        float sum = 0.0f;
        
        // Sum the elements in the column
        for (int row = 0; row < rows; row++) {
            sum += matrix[row * cols + col];
        }

        // Calculate the mean and store it in the result
        result[col] = sum / rows;
    }
}

__global__ void elementWiseSubtractKernel(float* A, const float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        C[idx] = A[idx] - B[idx];
    }
}
