#include <iostream>
#include <xtensor/xarray.hpp>
#include <nlohmann/json.hpp>
#include <xtensor/xadapt.hpp>
using namespace std;
using namespace xt;

float *cast_xarray(const xarray<float> xarr, bool transpose)
{   
    int rows = xarr.shape(0);
    int cols = xarr.shape(1);

    // Allocate memory for the new float array
    float *x_ptr = new float[rows * cols];

    // Fill the x_ptr array
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Access the element at (i, j) and assign it to x_ptr
            x_ptr[i * cols + j] = transpose ? xarr(j, i) : xarr(i, j);
        }
    }

    return x_ptr;
}

void allocateAndCopyToDevice(float* matrix_ptr, int rows, int cols, float** device_matrix) {
    // Calculate memory size based on the rows and columns
    unsigned int mem_size = sizeof(float) * rows * cols;

    // Allocate memory on the GPU
    cudaMalloc((void**)device_matrix, mem_size);

    // Copy data from host to device
    cudaMemcpy(*device_matrix, matrix_ptr, mem_size, cudaMemcpyHostToDevice);
}