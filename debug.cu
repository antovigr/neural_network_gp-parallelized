#include "src/cuda/matrix_product.cu"
#include "src/include/_debug.hpp"
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include "src/include/utils.cuh"
#include "src/cuda/utils.cu"
using namespace std;
using namespace xt;
#include <typeinfo>


int main()
{

	xt::xarray<float> a = {{1., 2., 3.}, {3., 4., 5.}};
	float* b = cast_xarray(a);

	print_matrix(b, 2, 3);

	return 0;
}