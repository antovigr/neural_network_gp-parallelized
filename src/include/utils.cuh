#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>
#include <xtensor/xarray.hpp>
#include <nlohmann/json.hpp>
#include <xtensor/xadapt.hpp>
using namespace std;
using namespace xt;

/**
 * @brief Copy the values of an xtensor array into a plain C array of floats.
 * 
 * This function allocates memory for a new C array and copies the elements 
 * from the given xtensor array. Optionally, the array can be transposed 
 * during the process.
 * 
 * @param xarr The xtensor array of floats to be copied.
 * @param transpose If true, the array is transposed before copying.
 * @return A pointer to the newly allocated C array of floats.
 *         The caller is responsible for freeing this memory.
 */
float* cast_xarray(const xarray<float> xarr, bool transpose);
void allocateAndCopyToDevice(float* matrix_ptr, int rows, int cols, float** device_matrix);


#endif