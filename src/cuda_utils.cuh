#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

class CudaMatrixMemory
{
public:
    // Constructor
    CudaMatrixMemory(const int rows, const int cols);

    // Class members
    float *device_ptr;
    unsigned int memory_size;
    
private:
};


#endif