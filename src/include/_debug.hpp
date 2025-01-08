#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <iostream>

void print_matrix(float* matrix, int N) {
    // Print result
    for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			std::cout<<matrix[i+j*N]<< " ";
		}
		std::cout<<std::endl;
	}
}

#endif