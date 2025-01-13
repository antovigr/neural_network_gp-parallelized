#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

CudaMatrixMemory::CudaMatrixMemory(const int rows, const int cols) : rows(rows), cols(cols) {
    memory_size = sizeof(float) * rows * cols;
    cudaError_t err = cudaMalloc((void**)&device_ptr, memory_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(err) << std::endl;
    }
};

CudaMatrixMemory::~CudaMatrixMemory() {
    cudaFree(device_ptr);
}

void CudaMatrixMemory::sendMatrix2Device(const float *carray) {
    cudaMemcpy(device_ptr, carray, memory_size, cudaMemcpyHostToDevice);
}

float* CudaMatrixMemory::allocAndSend2Host() {
    // Allocate memory for the host
    float* host_ptr = new float[rows * cols]; // Use new[] for proper cleanup with delete[]
    
    if (host_ptr == nullptr) { // Check for successful allocation
        throw std::runtime_error("Memory allocation failed on host.");
    }
    
    // Copy data from device to host
    cudaError_t err = cudaMemcpy(host_ptr, device_ptr, memory_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        delete[] host_ptr;  // Cleanup in case of failure
        throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }

    return host_ptr;
}

void CudaGrid::setKernelGrid(const int blocksize_x, const int blocksize_y, const int rows, const int cols) {
    threads = dim3(blocksize_x, blocksize_y);
    grid = dim3((cols + blocksize_x - 1) / blocksize_x, (rows + blocksize_y - 1) / blocksize_y);
}