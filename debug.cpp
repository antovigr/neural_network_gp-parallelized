#include "src/cuda/matrix_product.cu"

using namespace std;


int main() {

    c = matrix_product()

    // Print result
    for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			std::cout<<c[i+j*N]<< " ";
		}
		std::cout<<std::endl;
	}

    return 0
}