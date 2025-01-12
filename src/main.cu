#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>
#include <nlohmann/json.hpp> // Include the nlohmann/json library
#include "src/objects/model.cpp"
#include "src/objects/gradient_descent.cu"
#include "src/objects/autoencoder.cpp"
#include "src/utils.cpp"
#include "src/cuda/utils.cu"
#include "src/cuda/linear_algrebra.cu"

using namespace std;
using namespace xt;

void print_help(const string &program_name)
{
    std::cout << "Usage: " << program_name << " config.json\n";
    std::cout << "Pass a config.json file containing parameters.\n";
}

int main(int argc, char *argv[]) {

    // Check if the --help argument is passed
    if (argc > 1 && string(argv[1]) == "--help")
    {
        print_help(argv[0]);
        return 0;
    }

    // Send error message
    if (argc < 2)
    {
        cerr << "Error: Missing config.json file path.\n";
        print_help(argv[0]);
        return 1;
    }

    // Parsed the config_file argument
    string config_file_path = argv[1];

    // Read the JSON file
    nlohmann::json config = read_json(config_file_path);

    // Parse the arguments from the JSON file
    string dataset_path = config["dataset_path"];
    vector<int> network_architecture = config["network_architecture"];
    unsigned int epochs = config["epochs"];
    int batch_size = config["batch_size"];
    float learning_rate = config["learning_rate"];
    float train_test_split = config["train_test_split"];

    // Load dataset
    xt::xarray<double> x_train, y_train, x_test, y_test;
    if (dataset_path == "mnist") {
        std::tie(x_train, y_train, x_test, y_test) = Autoencoder::load_mnist_dataset(train_test_split);

    } else {
        ifstream infile(dataset_path);
        check_iostream_state(infile, dataset_path);
        xt::xarray<double> dataset = xt::load_csv<double>(infile);
        infile.close();

        int train_test_split_idx = round(train_test_split * dataset.shape(0));
        auto train_range = xt::range(0, train_test_split_idx);
        auto test_range = xt::range(train_test_split_idx, dataset.shape(0));
        x_train = xt::view(dataset, train_range, xt::range(0, dataset.shape(1))); // - 1
        y_train = xt::view(dataset, train_range, dataset.shape(1) - 1);
        x_test = xt::view(dataset, test_range, xt::range(0, dataset.shape(1))); // - 1
        y_test = xt::view(dataset, test_range, dataset.shape(1) - 1);
    }
    
    // Scale data
    scale_data(x_train);
    scale_data(y_train);
    scale_data(x_test);
    scale_data(y_test);

    // Load model
    Autoencoder nn(network_architecture, x_train.shape(1));
    // Args are network architecture and input size of the data 

    // Load pretrained weights if desired
    if (config["pretrained_model_path"].is_null()) {
        // Do nothing, randomly initialized weights will stay still
    } else {
        // Load pretrained weights and loss 
        string pretrained_model_path = config["pretrained_model_path"];
        nn.load_weights(pretrained_model_path);
        nn.load_loss(pretrained_model_path);
    }

    // Train model
    nn.fit(x_train, epochs, batch_size, learning_rate);
    
    // Save trained weights and loss if desired
    if (!config["model_save_path"].is_null()) {
        string model_save_path = config["model_save_path"];
        try {
            nn.save_weights(model_save_path);
            nn.save_loss(model_save_path);
        } catch (const std::runtime_error& e) {
            std::cout << "The program failed to save the model in " << model_save_path << ", saving in the current directory instead.";
            nn.save_weights(".");
            nn.save_loss(".");
        }
    }

    // Predict test set and save result if desired
    if (!config["pred_save_path"].is_null()) {
        // Predict
        xt:xarray<double> y_pred = nn.predict(x_train);
        // Save
        string pred_save_path = config["pred_save_path"];
        std::ofstream out_file (pred_save_path);
        xt::dump_csv(out_file, y_pred);
    }
    return 0;
}