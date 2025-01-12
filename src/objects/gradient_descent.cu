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
#include <cuda.h>
#include <typeinfo>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


using namespace std;
using namespace xt;

// Define constructor
// Just init class members
GradientDescent::GradientDescent(const xarray<double> &x_train, const xarray<double> &y_train, vector<xarray<double>> &weights, vector<xarray<double>> &biases) : x_train(x_train), y_train(y_train), weights(weights), biases(biases) {
    num_layers = weights.size(); 
    layer_outputs.resize(num_layers);
    layer_activations.resize(num_layers + 1);

    // Prepare data for cuda
    float *x_train_ptr = cast_xarray(x_train, true);
    float *y_train_ptr = cast_xarray(y_train, false);

    float *device_xtrain;
    float *device_ytrain;
    allocateAndCopyToDevice(x_train_ptr, x_train.shape(0), x_train.shape(1), &device_xtrain);
    allocateAndCopyToDevice(y_train_ptr, y_train.shape(0), y_train.shape(1), &device_ytrain);
    // device_x, y are cuda allocated matrix, x is transposed

    delete[] x_train_ptr;
    delete[] y_train_ptr;

    // Allocate and copy each layer's data to the device (they are just initialized at 0 for now)
    for (size_t i = 0; i < layer_outputs.size(); ++i) {
        float* layer_output_ptr = cast_xarray(layer_outputs[i], false);
        float* layer_activation_ptr = cast_xarray(layer_activations[i], false);

        float* device_layer_output;
        float* device_layer_activation;

        allocateAndCopyToDevice(layer_output_ptr, layer_outputs[i].shape(0), layer_outputs[i].shape(1), &device_layer_output);
        allocateAndCopyToDevice(layer_activation_ptr, layer_activations[i].shape(0), layer_activations[i].shape(1), &device_layer_activation);

        // Store device pointers
        device_layer_outputs.push_back(device_layer_output);
        device_layer_activations.push_back(device_layer_activation);

        // Clean up host memory after copying to device
        delete[] layer_output_ptr;
        delete[] layer_activation_ptr;
    }
}


// Write gradient descent methods

void GradientDescent::forward_pass(const xarray<double> &x_batch) {

    layer_activations[0] = xt::transpose(x_batch);

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
    int dataset_size = x_train.shape()[0];
    int batch_number = (dataset_size / batch_size);

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

            // Perform the forward pass
            forward_pass(x_batch); // Modify the layer_activations and layer_outputs
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

