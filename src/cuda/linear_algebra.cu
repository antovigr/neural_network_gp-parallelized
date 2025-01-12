#include <iostream>
#include <cuda_runtime.h>

__global__ void transposeKernel(float* d_input, float* d_output, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        int input_idx = j * width + i;  // 1D index of element in the input matrix
        int output_idx = i * height + j;  // 1D index of element in the output matrix
        d_output[output_idx] = d_input[input_idx];
    }
}