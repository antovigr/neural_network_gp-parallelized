#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <optional>

using namespace std;
using namespace xt;

class GradientDescent
{
public:
    // Constructor
    GradientDescent(const xarray<double>& x_train, const xarray<double>& y_train, 
                    vector<xarray<double>>& weights, vector<xarray<double>>& biases);
    
    // Method to start training
    void train(const unsigned int& epochs, const int& batch_size, const float& learning_rate);

    // Class members
    vector<xarray<double>> weights;   // Weights of the network
    vector<xarray<double>> biases;    // Biases of the network
    vector<double> loss_history;      // History of loss over epochs
    xarray<double> x_train;           // Training data (inputs)
    xarray<double> y_train;           // Labels corresponding to the training data

private:
    // Forward pass through the network
    void forward_pass(float* device_xbatch_ptr,float *device_xbatch_transpose, int &current_batch_size, int &width);

    // Backward pass to calculate gradients and update weights/biases
    void backward_pass(const xarray<double>& y_batch, const int& current_batch_size, const float& learning_rate);

    void train_allocate_device_memory(int &batch_size);


    // Declare class members

    int num_layers;
    
    // Declare class member device pointers allocated in the constructor
    float *device_xtrain;
    float *device_ytrain;
    float *device_xtrainT;
    float *device_ytrainT;
    vector<float*> device_weights;
    vector<float*> device_biases;
    vector<float*> device_gradient_w;
    vector<float*> device_gradient_b;

    // Declare class member device pointers allocated in the train method (dependent on the batch size)
    float *device_xbatch;
    vector<float*> device_l_o; // l_o stands for layer_outputs
    vector<float*> device_l_a; // l_a stands for layer_activations
    vector<float*> device_deltas;
};

#endif // GRADIENT_DESCENT_HPP
