#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>
#include <xtensor/xarray.hpp>
#include <nlohmann/json.hpp>
#include <xtensor/xadapt.hpp>
using namespace std;
using namespace xt;

float* cast_xarray(const xarray<float> xarr, bool transpose);

#endif