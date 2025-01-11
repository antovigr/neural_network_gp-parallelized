#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <iostream>

void print_matrix(float* matrix, int rows, int cols) {
    // Print result
    for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			std::cout<<matrix[i * cols + j]<< " ";
		}
		std::cout<<std::endl;
	}
}

#endif