#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

#include "model.hpp"
#include <xtensor/xarray.hpp>
#include <iostream>

using namespace xt;

class Autoencoder : public Model {
public:
    // Constructor
    Autoencoder(const std::vector<int>& architecture, const int& input_size)
        : Model(architecture, input_size) {};  // Call the base class parameterized constructor

    // Methods
    void fit(const xarray<float> &x_train, const unsigned int &epochs, const int &batch_size, const float &learning_rate);
    static tuple<xarray<float>, xarray<float>, xarray<float>, xarray<float>> load_mnist_dataset(const float &train_test_split);
};

#endif