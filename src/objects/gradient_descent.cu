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

    // Transpose xtrain and ytrain on device
    unsigned int mem_size_xtrainT = sizeof(float) * x_train.shape(0) * x_train.shape(1);
    unsigned int mem_size_ytrainT = sizeof(float) * y_train.shape(0) * y_train.shape(1);
    cudaMalloc((void**)device_xtrainT, mem_size_xtrainT);
    cudaMalloc((void**)device_ytrainT, mem_size_ytrainT);
    int blocksize = 16;

    int width = x_train.shape(0);
    int height = x_train.shape(1);
    dim3 threads(blocksize, blocksize);
    dim3 grid((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
    transposeKernel<<<grid, threads>>>(device_xtrain, device_xtrainT, width, height);

    int width = y_train.shape(0);
    int height = y_train.shape(1);
    dim3 threads(blocksize, blocksize);
    dim3 grid((width + blocksize - 1) / blocksize, (height + blocksize - 1) / blocksize);
    transposeKernel<<<grid, threads>>>(device_ytrain, device_ytrainT, width, height);

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

        // Allocate memory
        allocateAndCopyToDevice(w_i, weights[i].shape(0), weights[i].shape(1), &d_w_i);
        allocateAndCopyToDevice(b_i, biases[i].shape(0), biases[i].shape(1), &d_b_i);
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

        // Clean up host memory after copying to device
        delete[] w_i;
        delete[] b_i;
    }
    delete[] x_train_ptr;
    delete[] y_train_ptr;
}


// Write gradient descent methods

void GradientDescent::forward_pass(float* device_xbatch_ptr, float *device_xbatch_transpose, int &current_batch_size, int &width) {

    transposeKernel(device_xbatch_ptr, device_xbatch_transpose, width, current_batch_size);

    for (size_t l = 0; l < num_layers; l++) {
        layer_outputs[l] = xt::linalg::dot(weights[l], layer_activations[l]) + biases[l];
        layer_activations[l + 1] = sigmoid(layer_outputs[l]); // sigmoid is defined in src/utils/utils.cpp
    }
}

void GradientDescent::backward_pass(const xarray<double> &y_batch, const int &current_batch_size, const float &learning_rate) {
    
    vector<xarray<double>> deltas(num_layers);

    // Init delta vector corresponding to the last layer
    xarray<double> &last_activation = layer_activations[num_layers];
    deltas[num_layers - 1] = (last_activation - xt::transpose(y_batch)) * sigmoid_derivative(layer_outputs[num_layers - 1]);

    for (int l = num_layers - 2; l >= 0; l--) {
        deltas[l] = xt::linalg::dot(xt::transpose(weights[l + 1]), deltas[l + 1]) * sigmoid_derivative(layer_outputs[l]);
    }

    // Update weights and biases
    for (int l = 0; l < num_layers; l++) {
        xarray<double> gradient_w = xt::linalg::dot(deltas[l], xt::transpose(layer_activations[l])) / current_batch_size; // Batch size may vary, at the end of epoch
        xarray<double> gradient_b = xt::mean(deltas[l], {1});
        gradient_b = gradient_b.reshape({gradient_b.size(), 1});

        weights[l] -= learning_rate * gradient_w;
        biases[l] -= learning_rate * gradient_b;
    }
}

void GradientDescent::train(const unsigned int &epochs, const int &batch_size, const float &learning_rate) {

    train_allocate_device_memory(batch_size); // Work with fixed current_batch_size

    int dataset_size = x_train.shape()[0];
    int batch_number = (dataset_size / batch_size);

    // Allocate device memory for storing transposed x_batch matrix
    // TODO: remove this step by directly casting transposed matrix in constructor
    float *device_xbatch_transpose;
    unsigned int device_mem_size = sizeof(float) * batch_size * x_train.shape(1); // Number of observations * number of features
    cudaMalloc((void**)device_xbatch_transpose, device_mem_size);
    
    for (unsigned int epoch = 0; epoch < epochs; epoch++) {

        cout << "Epoch: " << epoch << endl;
        float epoch_mse = 0;
        int batch_id = 0;

        for (int batch_start = 0; batch_start < dataset_size; batch_start += batch_size) {
            // Plot currently processed batch number and increment
            cout << "   Batch: " << batch_id << " / " << batch_number << endl;
            batch_id++;

            // Compute the current batch size, as defined normally but smaller if at the end of epoch
            int current_batch_size = std::min(batch_size, dataset_size - batch_start);
            xarray<double> x_batch = xt::view(x_train, range(batch_start, batch_start + current_batch_size), all());
            xarray<double> y_batch = xt::view(y_train, range(batch_start, batch_start + current_batch_size), all());
            // Create device sub array for the batch by using pointer arithmetic
            int batch_idx_start = batch_start * x_train.shape(1); // Multiply by the columns number for getting the right index of row
            float *device_xbatch_ptr = device_xtrain + batch_idx_start;
            // int batch_idx_stop = batch_idx_start + current_batch_size * x_train.shape(1); // Add the right number of rows

            // Perform the forward pass
            // Specify width of the xbatch matrix for the transposition
            int xbatch_width = x_train.shape(1); // i.e. the number of features
            // TODO: make a DeviceMatrix class for cleaning this process
            forward_pass(device_xbatch_ptr, device_xbatch_transpose, current_batch_size, xbatch_width); // Modify the layer_activations and layer_outputs
            xarray<double> &last_activation = layer_activations[num_layers];

            // Perform the backward pass
            backward_pass(y_batch,  current_batch_size, learning_rate); // Modify the weights and biases
 
            // Compute the loss for the current batch
            xarray<double> squared_error = xt::pow(last_activation - xt::transpose(y_batch), 2); // Error for each pixel of each observation
            xarray<double> observation_mse = xt::mean(squared_error, {0}); // Mean over all the pixels in the observations
            epoch_mse += xt::sum(observation_mse)() / dataset_size;
        }
        cout << "   MSE: " << epoch_mse << endl;
        loss_history.push_back(epoch_mse);
    }
}

void GradientDescent::train_allocate_device_memory(int &batch_size) {
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

}

