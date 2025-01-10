#include "src/cuda/matrix_product.cu"
#include "src/include/_debug.hpp"

using namespace std;
#include <typeinfo>

int main()
{

	int rowsA = 16;
	int colsA = 16;
	int rowsB = 16;
	int colsB = 16;
	int N = rowsA; // Square matrices
	int BLOCK_SIZE = 4;

	float *matA = new float[rowsA * colsA];
	float *matB = new float[rowsB * colsB];
	float *matC = new float[rowsA * colsB];

	// Fill matrices
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			matA[i + j * N] = i;
			matB[i + j * N] = j;
		}
	}

	float *_ = matrix_product_cuda(matA, rowsA, colsA, matB, rowsB, colsB, matC, BLOCK_SIZE);

	return 0;
}