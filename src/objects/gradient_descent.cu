#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <vector>
#include <string>
#include "src/include/gradient_descent.hpp"
#include "src/include/utils.hpp"

// Cuda utils
#include "src/include/gemm_kernel.cuh"
#include "src/include/matrix_utils.h"
#include "src/include/utils.cuh"
#include "src/include/linear_algebra.cuh"
#include <cuda.h>
#include <typeinfo>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


using namespace std;
using namespace xt;

// Define constructor
// Just init class members
GradientDescent::GradientDescent(const xarray<double> &x_train, const xarray<double> &y_train, vector<xarray<double>> &weights, vector<xarray<double>> &biases) : x_train(x_train), y_train(y_train), weights(weights), biases(biases) {
    // Compute the number of layers
    num_layers = weights.size(); 

    // Allocate device memory
    float *x_train_ptr = cast_xarray(x_train, false);
    float *y_train_ptr = cast_xarray(y_train, false);

    allocateAndCopyToDevice(x_train_ptr, x_train.shape(0), x_train.shape(1), &device_xtrain);
    allocateAndCopyToDevice(y_train_ptr, y_train.shape(0), y_train.shape(1), &device_ytrain);
    // device_xtrain, device_ytrain are cuda allocated matrix, x is not transposed


    // Cast weights and biases for each layer and allocate memory to device
    for (size_t i = 0; i < num_layers; ++i) {
        // Cast xarrays stored in vectors
        float* w_i = cast_xarray(weights[i], false);
        float* b_i = cast_xarray(biases[i], false);

        // Create device pointer
        float* d_w_i; // Weights
        float* d_b_i;
        float* d_g_w_i; // Gradient of weights
        float* d_g_b_i;
        float* d_wT_i;
        float* d_bT_i;
    

        // Allocate memory
        allocateAndCopyToDevice(w_i, weights[i].shape(0), weights[i].shape(1), &d_w_i);
        allocateAndCopyToDevice(b_i, biases[i].shape(0), biases[i].shape(1), &d_b_i);
        allocateAndCopyToDevice(w_i, weights[i].shape(0), weights[i].shape(1), &d_wT_i); // trick to allocate memory for transposition of gradient
        allocateAndCopyToDevice(b_i, biases[i].shape(0), biases[i].shape(1), &d_bT_i);
        // Allocate memory without copying
        unsigned int d_g_w_i_size = sizeof(float) * weights[i].shape(0) * weights[i].shape(1);
        unsigned int d_g_b_i_size = sizeof(float) * biases[i].shape(0) * biases[i].shape(1);
        cudaMalloc((void**)d_g_w_i, d_g_w_i_size);
        cudaMalloc((void**)d_g_b_i, d_g_b_i_size);

        // Store device pointers into vectors
        device_weights.push_back(d_w_i);
        device_biases.push_back(d_b_i);
        device_gradient_w.push_back(d_g_w_i);
        device_gradient_b.push_back(d_g_b_i);
        device_weightsT.push_back(d_wT_i);
        device_biasesT.push_back(d_bT_i);

        // Clean up host memory after copying to device
        delete[] w_i;
        delete[] b_i;
    }
    delete[] x_train_ptr;
    delete[] y_train_ptr;
}


// Write gradient descent methods

void GradientDescent::forward_pass(float* device_xbatch_ptr, int &current_batch_size) {
    
    // Init activations
    // width height are for the input matrix to put in kernel
    int blocksize = 16; 
    int height = current_batch_size; // nb of observations in the batch
    int width = x_train.shape(1); // nb of features 
    dim3 threads(blocksize, blocksize);
    dim3 grid((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
    transposeKernel<<<grid, threads>>>(device_xbatch_ptr, device_l_a[0], width, height);

    for (size_t l = 0; l < num_layers; l++) {
        // layer_output[l]
        // layer_outputs[l] = xt::linalg::dot(weights[l], layer_activations[l]) + biases[l];
        // dot product 
        height = weights[l].shape(0); // nb of neurons
        width = current_batch_size; // nb of observations
        int K = l > 0 ? weights[l - 1].shape(0) : x_train.shape(1); // nb of neurons of precedent layer or nb of features
        blocksize = 16; 
        threads = dim3(blocksize, blocksize);
        grid = dim3((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
        gemmKernel<<<grid, threads>>>(device_weights[l], device_l_a[l], device_l_o[l], height, width, K);
        // add
        addBiasKernel<<<grid, threads>>>(device_l_o[l], device_biases[l], device_l_o[l], height, width);

             
        // layer_activations[l + 1] = sigmoid(layer_outputs[l]);
        sigmoidKernel<<<grid, threads>>>(device_l_o[l], device_l_a[l + 1], width * height);

    }
}

void GradientDescent::backward_pass(float* device_ybatch_ptr, const int& current_batch_size, const float& learning_rate) {
    // Transpose y_batch
    int blocksize = 16; 
    int height = current_batch_size; // nb of observations in the batch
    int width = y_train.shape(1); // nb of features 
    dim3 threads(blocksize, blocksize);
    dim3 grid((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
    transposeKernel<<<grid, threads>>>(device_ybatch_ptr, device_ybatchT, width, height);


    // Init delta vector corresponding to the last layer
    // xarray<double> &last_activation = layer_activations[num_layers];
    // deltas[num_layers - 1] = (last_activation - xt::transpose(y_batch)) * sigmoid_derivative(layer_outputs[num_layers - 1]);
    // substract
    height = y_train.shape(1); // nb of features
    width = current_batch_size; // nb of observations in the batch
    blocksize = 16; 
    threads = dim3(blocksize, blocksize);
    grid = dim3((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
    subtractKernel<<<grid, threads>>>(device_l_a[num_layers], device_ybatchT, device_deltas[num_layers - 1], height, width);
    // apply sigmoid derivative
    sigmoidDerivativeKernel<<<grid, threads>>>(device_l_o[num_layers - 1], device_l_o[num_layers - 1], width * height);
    // multiply element wise
    elementWiseMultiply<<<grid, threads>>>(device_deltas[num_layers -1], device_l_o[num_layers - 1], device_deltas[num_layers -1], height * width);


    for (int l = num_layers - 2; l >= 0; l--) {
        // deltas[l] = xt::linalg::dot(xt::transpose(weights[l + 1]), deltas[l + 1]) * sigmoid_derivative(layer_outputs[l]);

        // Perform matrix multiplication: deltas[l] = transpose(weights[l + 1]) * deltas[l + 1]
        // transpose
        width = weights[l + 1].shape(1); 
        height = weights[l + 1].shape(0);
        blocksize = 16;
        threads= dim3(blocksize, blocksize);
        grid = dim3((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
        transposeKernel<<<grid, threads>>>(device_weights[l + 1], device_weightsT[l + 1], width, height);

        // dot product
        height = weights[l + 1].shape(1);
        width = current_batch_size; 
        int K = weights[l + 1].shape(0);
        blocksize = 16;
        threads = dim3(blocksize, blocksize);
        grid = dim3((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
        gemmKernel<<<grid, threads>>>(device_weightsT[l + 1], device_deltas[l + 1], device_deltas[l], height, width, K);
        
        // Apply the sigmoid derivative to the activation of the current layer
        sigmoidDerivativeKernel<<<grid, threads>>>(device_l_o[l], device_l_o[l], width * height);
        
        // Multiply element-wise: deltas[l] = deltas[l] * sigmoid_derivative(l_a[l])
        elementWiseMultiply<<<grid, threads>>>(device_deltas[l], device_l_o[l], device_deltas[l], height * width);
    }

    // Update weights and biases
    for (int l = 0; l < num_layers; l++) {
        // gradient_w = xt::linalg::dot(deltas[l], xt::transpose(layer_activations[l])) / current_batch_size; // Batch size may vary, at the end of epoch
        // dot product
        int Arows = weights[l].shape(0);
        int Acols = current_batch_size;
        int Brows = l > 1 ? weights[l - 1].shape(0) : x_train.shape(1);
        int Bcols = current_batch_size;
        height = Arows;
        width = Bcols; 
        blocksize = 16;
        threads = dim3(blocksize, blocksize);
        grid = dim3((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
        matrixMultiplyTransposeKernel<<<grid, threads>>>(device_deltas[l], device_l_a[l], device_gradient_w[l], Arows, Acols, Brows, Bcols);
        
        // division 
        float scalar = 1 / current_batch_size;
        elementWiseMultiplyScalarKernel<<<grid, threads>>>(device_gradient_w[l], scalar, device_gradient_w[l], height, width);

        // gradient_b = xt::mean(deltas[l], {1});
        // gradient_b = gradient_b.reshape({gradient_b.size(), 1});
        width = Acols;
        height = Arows; 
        blocksize = 16;
        threads = dim3(blocksize, blocksize);
        grid = dim3((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
        matrixMeanRowKernel<<<grid, threads>>>(device_deltas[l], device_gradient_b[l], height, width);

        // Update weights: weights[l] -= learning_rate * gradient_w
        width = weights[l].shape(1);
        height = weights[l].shape(0); 
        blocksize = 16;
        threads = dim3(blocksize, blocksize);
        grid = dim3((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
        elementWiseMultiplyScalarKernel<<<grid, threads>>>(device_gradient_w[l], learning_rate, device_gradient_w[l], height, width);
        elementWiseSubtractKernel<<<grid, threads>>>(device_weights[l], device_gradient_w[l], device_weights[l], height, width); 

        // Update biases: biases[l] -= learning_rate * gradient_b
        width = biases[l].shape(1);
        height = biases[l].shape(0); 
        blocksize = 16;
        threads =dim3(blocksize, blocksize);
        grid =dim3((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
        elementWiseMultiplyScalarKernel<<<grid, threads>>>(device_gradient_b[l], learning_rate, device_gradient_b[l], height, width);
        elementWiseSubtractKernel<<<grid, threads>>>(device_biases[l], device_gradient_b[l], device_biases[l], height, width); 

    }
}

void GradientDescent::train(const unsigned int &epochs, const int &batch_size, const float &learning_rate) {

    train_allocate_device_memory(batch_size); // Work with fixed current_batch_size

    int dataset_size = x_train.shape()[0];
    int batch_number = (dataset_size / batch_size);
    
    for (unsigned int epoch = 0; epoch < epochs; epoch++) {

        cout << "Epoch: " << epoch << endl;
        float epoch_mse = 0;
        int batch_id = 0;

        for (int batch_start = 0; batch_start < dataset_size; batch_start += batch_size) {
            if (batch_size <= dataset_size - batch_start) {
                break;
            }
            // Plot currently processed batch number and increment
            cout << "   Batch: " << batch_id << " / " << batch_number << endl;
            batch_id++;

            // Compute the current batch size, as defined normally but smaller if at the end of epoch
            int current_batch_size = batch_size;
            // Create device sub array for the batch by using pointer arithmetic
            int batch_idx_start = batch_start * x_train.shape(1); // Multiply by the columns number for getting the right index of row
            float *device_xbatch_ptr = device_xtrain + batch_idx_start;
            // int batch_idx_stop = batch_idx_start + current_batch_size * x_train.shape(1); // Add the right number of rows

            // Perform the forward pass
            forward_pass(device_xbatch_ptr, current_batch_size); // Modify the layer_activations and layer_outputs

            // Perform the backward pass
            int batch_idx_start_y = batch_start * y_train.shape(1);
            float *device_ybatch_ptr = device_ytrain + batch_idx_start_y;
            backward_pass(device_ybatch_ptr,  current_batch_size, learning_rate); // Modify the weights and biases
 
            // Compute the loss for the current batch
            // xarray<double> &last_activation = layer_activations[num_layers];
            // xarray<double> squared_error = xt::pow(last_activation - xt::transpose(y_batch), 2); // Error for each pixel of each observation
            // xarray<double> observation_mse = xt::mean(squared_error, {0}); // Mean over all the pixels in the observations
            // epoch_mse += xt::sum(observation_mse)() / dataset_size;
        }
        // cout << "   MSE: " << epoch_mse << endl;
        // loss_history.push_back(epoch_mse);
    }
}

void GradientDescent::train_allocate_device_memory(const int &batch_size) {
    // Declare the fixed size of the vectors
    device_l_o.resize(num_layers);
    device_l_a.resize(num_layers + 1);
    device_deltas.resize(num_layers);

    // Allocate memory iteratively without copying
        
    // Allocate activations[0]
    unsigned int mem_size_l_a = sizeof(float) * x_train.shape(1) * batch_size; // number of features (pixels) * batch_size
    cudaMalloc((void**)device_l_a[0], mem_size_l_a);
    
    for (size_t i = 0; i < num_layers; ++i) {

        // Create device pointer
        float* d_l_o; 
        float* d_l_a;

        // Allocate memory without copying
        unsigned int mem_size_l_o = sizeof(float) * weights[i].shape(0) * batch_size;
        unsigned int &mem_size_l_a = mem_size_l_o;

        cudaMalloc((void**)d_l_o, mem_size_l_o);
        cudaMalloc((void**)d_l_a, mem_size_l_a);

        // Store device pointers into vectors
        device_l_o[i] = d_l_o;
        device_l_a[i + 1] = d_l_a;
   
    }

    // Allocate memory for deltas going backward
    unsigned int mem_size_delta = sizeof(float) * weights[num_layers - 1].shape(0) * batch_size;
    cudaMalloc((void**)device_deltas[num_layers - 1], mem_size_delta);    

    for (int l = num_layers - 2; l >= 0; l--) {
        // deltas[l] = xt::linalg::dot(xt::transpose(weights[l + 1]), deltas[l + 1]) * sigmoid_derivative(layer_outputs[l]);
        
        float* d_delta;
        
        unsigned int mem_size_delta = sizeof(float) * weights[l + 1].shape(1) * batch_size;

        cudaMalloc((void**)d_delta, mem_size_delta);

        device_deltas[l] = d_delta;
    }
    unsigned int mem_y_batch = sizeof(float) * batch_size * y_train.shape(1);
    cudaMalloc((void**)device_ybatchT, mem_y_batch);

}

