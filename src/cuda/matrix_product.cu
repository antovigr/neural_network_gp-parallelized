#include "src/include/gemm_kernel.cuh"
#include "src/include/gemm_noblas.h"
#include "src/include/matrix_utils.h"

using namespace std;


vector<float> matrix_product_cuda(vector<float> &matrixA, int& rowsA, int& colsA, vector<float> &matrixB, int& rowsB, int& colsB) {

    int& WA = colsA;
    int& HA = rowsA;
    int& WB = colsB;
    int& HB = rowsB;

    // utilities
    cudaEvent_t start;
    cudaEvent_t stop;

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // initialize host memory
    fill_random<REAL>(h_A, WA, HA);
    fill_random<REAL>(h_B, WB, HB);

    // allocate device memory
    float *d_A;
    cudaMalloc((void **)&d_A, mem_size_A);
    float *d_B;
    cudaMalloc((void **)&d_B, mem_size_B);

    // allocate device memory for result
    unsigned int size_C = WA * HB;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *d_C;
    cudaMalloc((void **)&d_C, mem_size_C);

    // allocate host memory for the result
    float *h_C = (float *)malloc(mem_size_C);

    dim3 threads, grid;

    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    // setup execution parameters
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    grid = dim3(WC / threads.x, HC / threads.y);

    // execute the kernel
    gemm_naive<<<grid, threads>>>(d_C, d_A, d_B, WA, WB);

    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

}
