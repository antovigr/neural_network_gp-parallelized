#include <iostream>
#include <xtensor/xarray.hpp>
#include <nlohmann/json.hpp>
#include <xtensor/xadapt.hpp>
using namespace std;
using namespace xt;

float* cast_xarray(const xarray<float> xarr)
{
    // Copy the values of an xtensor array into a plain C array of floats

    int rows = xarr.shape(0);
    int cols = xarr.shape(1);

    // Allocate memory for the new float array
    float* x_ptr = new float[rows * cols];

    // Fill the x_ptr array
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Access the element at (i, j) and assign it to x_ptr
            x_ptr[i * cols + j] = xarr(i, j);  // Direct access to the element
        }
    }

    return x_ptr;
}