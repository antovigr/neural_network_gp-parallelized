#include "src/cuda/matrix_product.cu"

using namespace std;
#include <typeinfo>


int main() {

	vector<float> matA = {2, 2, 2, 2};
	vector<float> matB = {1, 1, 1, 2};

	int rowsA = 2;
	int colsA = 2;
	int rowsB = 2;
	int colsB = 2;
	int BLOCK_SIZE =1;
	int N = 2;

    float* matC = matrix_product_cuda(matA, rowsA, colsA, matB, rowsB, colsB, BLOCK_SIZE);
	cout << matC[0] << endl; // Directly access the first element of the array    
	
	// Print result
    for (int i = 1; i < N+1; i++)
	{
		for (int j = 1; j < N+1; j++)
		{
			std::cout<<matC[i+j*N]<< " ";
		}
		std::cout<<std::endl;
	}

    return 0;
}