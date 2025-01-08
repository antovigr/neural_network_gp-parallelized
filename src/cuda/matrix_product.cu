#include "src/include/gemm_kernel.cuh"
#include "src/include/gemm_noblas.h"
#include "src/include/matrix_utils.h"
#include "src/include/_debug.hpp"

#include <cmath>
#include <iostream>
#include <string>

#include <cuda.h>

using namespace std;

float* matrix_product_cuda(float *matrixA, int &rowsA, int &colsA, float *matrixB, int &rowsB, int &colsB, float *matrixC, int& BLOCK_SIZE)
{

    int &WA = colsA;
    int &HA = rowsA;
    int &WB = colsB;
    int &HB = rowsB;
    int &WC = WA;
    int &HC = HB;
    float *h_A = matrixA;
    float *h_B = matrixB;
    float *h_C = matrixC;

    print_matrix(h_A, colsA);
    print_matrix(h_B, colsB);
    print_matrix(h_C, colsA);
      
    // allocate device memory
    float *d_A;
    float *d_B;
    float *d_C;
    
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    
    unsigned int size_C = HA * WB;
    unsigned int mem_size_C = sizeof(float) * size_C;
    
    cudaMalloc((void **)&d_A, mem_size_A);
    cudaMalloc((void **)&d_B, mem_size_B);
    cudaMalloc((void **)&d_C, mem_size_C);


    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    // setup execution parameters
    dim3 threads, grid;
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    grid = dim3(WC / threads.x, HC / threads.y);

    // execute the kernel
    gemm_naive<<<grid, threads>>>(d_C, d_A, d_B, WA, WB);

    cudaDeviceSynchronize();

    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    print_matrix(h_C, rowsA);

    return h_C;
}
