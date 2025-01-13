#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <vector>
#include <string>
#include "gradient_descent.cuh"
#include "utils.hpp"
#include "cuda_utils.cuh"
#include "debugging_utils.hpp"
#include "cuda_operations.cuh"

using namespace std;
using namespace xt;

// Define constructor
// Just init class members
GradientDescent::GradientDescent(const xarray<double> &x_train, const xarray<double> &y_train, vector<xarray<double>> &weights, vector<xarray<double>> &biases, const int batch_size) : x_train(x_train), y_train(y_train), weights(weights), biases(biases), batch_size(batch_size) {
    num_layers = weights.size(); 
    layer_outputs.resize(num_layers);
    layer_activations.resize(num_layers + 1);

    // Initialize cuda arrays (allocate memory)
    
    // Init first layer input, which is the transpose of x_batch
    // Note that the indexing of LA (layer_activations) is somehow décalé: LA_l is the input of the layer L and output of the layer l-1
    int larows = x_train.shape(1);
    int lacols = batch_size;
    CudaMatrixMemory InitLayerActivation(larows, lacols);
    cuda_layer_activations.push_back(InitLayerActivation);
    
    for (size_t l = 0; l < num_layers; l++) {
        // Weights
        int wrows = weights[l].shape(0);
        int wcols = weights[l].shape(1);
        CudaMatrixMemory LayerWeights(wrows, wcols);
        cuda_weights.push_back(LayerWeights);

        // Biases
        int brows = biases[l].shape(0);
        int bcols = biases[l].shape(1);
        CudaMatrixMemory LayerBiases(brows, bcols);
        cuda_biases.push_back(LayerBiases);

        // Layer output = W_l * LA_l + B_l
        int lorows = wrows;
        int locols = cuda_layer_activations[l].cols;
        CudaMatrixMemory LayerOutput(lorows, locols);
        cuda_layer_outputs.push_back(LayerOutput);

        // Layer activation = sigmoid( LO_{l-1} )
        // We are pushing the element l + 1 of the vector now (because of the initialization before the loop)
        int larows = lorows;
        int lacols = locols;
        CudaMatrixMemory LayerActivation(larows, lacols);
        cuda_layer_activations.push_back(LayerActivation);

        printCudaMatrixShapes(LayerWeights, "LayerWeights");
        printCudaMatrixShapes(LayerBiases, "LayerBiases");
        printCudaMatrixShapes(LayerOutput, "LayerOutput");
        printCudaMatrixShapes(LayerActivation, "LayerActivation");
    }  
}


// Write gradient descent methods

void GradientDescent::forward_pass(const xarray<double> &x_batch) {

    layer_activations[0] = xt::transpose(x_batch);
    
    // Transform xtarray into carray
    ArrayHandler XBATCH_T;
    XBATCH_T.cast_xtarray(layer_activations[0]);

    // Copy XBATCH_T into cuda_layer_activations[0] i.e. the network's input
    CudaMatrixMemory& network_input = cuda_layer_activations[0];
    network_input.sendMatrix2Device(XBATCH_T.carray);
    
    // Perform computations with cuda
    for (size_t l = 0; l < num_layers; l++) {
        CudaMatrixMemory& w = cuda_weights[l];
        CudaMatrixMemory& b = cuda_biases[l];
        CudaMatrixMemory& lo = cuda_layer_outputs[l];
        CudaMatrixMemory& la = cuda_layer_activations[l];        
        CudaMatrixMemory& la_next = cuda_layer_activations[l + 1];
        
        CudaGrid matMulGrid;
        CudaGrid addGrid;
        CudaGrid sigmoidGrid;
        matMulGrid.setKernelGrid(16, 16, w.rows, w.cols);
        addGrid.setKernelGrid(16, 16, w.rows, w.cols);
        sigmoidGrid.setKernelGrid(16, 16, la_next.rows, la_next.cols);

        cudaDeviceSynchronize();
        matrixMulKernel<<<matMulGrid.grid, matMulGrid.threads>>>(w.device_ptr, la.device_ptr, lo.device_ptr, w.rows, w.cols, la.cols); // w * la, write the result in lo
        cudaDeviceSynchronize();
        addBiasToMatrixKernel<<<addGrid.grid, addGrid.threads>>>(lo.device_ptr, b.device_ptr, lo.device_ptr, lo.rows, lo.cols);
        sigmoidKernel<<<sigmoidGrid.grid, sigmoidGrid.threads>>>(lo.device_ptr, la_next.device_ptr, la_next.rows, la_next.cols);
        cudaDeviceSynchronize();

        // Copy back the computations into the base pipeline
        float* w_host = w.allocAndSend2Host();
        float* b_host = b.allocAndSend2Host();
        float* lo_host = lo.allocAndSend2Host();
        float* la_host = la.allocAndSend2Host();
        // float* la_next_host = la_next.allocAndSend2Host();


    }


    for (size_t l = 0; l < num_layers; l++) {
        layer_outputs[l] = xt::linalg::dot(weights[l], layer_activations[l]) + biases[l];
        layer_activations[l + 1] = sigmoid(layer_outputs[l]); // sigmoid is defined in utils/utils.cpp
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

void GradientDescent::train(const unsigned int &epochs, const float &learning_rate) {
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

