#include "src/include/gemm_kernel.cuh"
#include "src/include/matrix_utils.h"
#include "src/include/_debug.hpp"

#include <cmath>
#include <iostream>
#include <string>

#include <cuda.h>

using namespace std;

float* matrix_product_cuda(float *matrixA, int &rowsA, int &colsA, float *matrixB, int &rowsB, int &colsB, float *matrixC, int& BLOCK_SIZE)
{

    print_matrix(matrixA, rowsA);
    print_matrix(matrixB, rowsB);

    int &WA = colsA;
    int &HA = rowsA;
    int &WB = colsB;
    int &HB = rowsB;

    if (WA != HB) {
        std::cerr << "Matrix dimensions are incompatible for multiplication!" << std::endl;
        return nullptr;
    }

    int &WC = WB;
    int &HC = HA;

    unsigned int size_A = WA * HA;
    unsigned int size_B = WB * HB;
    unsigned int size_C = WC * HC;

    unsigned int mem_size_A = sizeof(float) * size_A;
    unsigned int mem_size_B = sizeof(float) * size_B;
    unsigned int mem_size_C = sizeof(float) * size_C;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, mem_size_A);
    cudaMalloc((void **)&d_B, mem_size_B);
    cudaMalloc((void **)&d_C, mem_size_C);

    cudaMemcpy(d_A, matrixA, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matrixB, mem_size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, mem_size_C);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((WC + BLOCK_SIZE - 1) / BLOCK_SIZE, (HC + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_naive<<<grid, threads>>>(d_C, d_A, d_B, WA, WB);
    cudaDeviceSynchronize();

    cudaMemcpy(matrixC, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    print_matrix(matrixC, rowsA);

    return matrixC;
}