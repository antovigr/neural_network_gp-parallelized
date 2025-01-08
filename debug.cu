#include "src/cuda/matrix_product.cu"
#include "src/include/_debug.hpp"

using namespace std;
#include <typeinfo>

int main()
{

	int rowsA = 4;
	int colsA = 4;
	int rowsB = 4;
	int colsB = 4;
	int N = rowsA; // Square matrices
	int BLOCK_SIZE = 2;

	float *matA = new float[rowsA * colsA];
	float *matB = new float[rowsB * colsB];
	float *matC = new float[rowsA * colsB];

	// Fill matrices
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			matA[i + j * N] = 1;
			matB[i + j * N] = 1;
		}
	}

	float *_ = matrix_product_cuda(matA, rowsA, colsA, matB, rowsB, colsB, matC, BLOCK_SIZE);

	return 0;
}