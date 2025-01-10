#!/bin/bash

# Define the output config file name
CONFIG_FILE="config.json"

# Define the JSON content
JSON_CONTENT='{
    "dataset_path": "datasets/autoencoding/noisy_sinus.csv",
    "train_test_split": 1,
    "network_architecture": [4, 20],
    "epochs": 1,
    "batch_size": 32,
    "learning_rate": 0.1,
    "model_save_path": null,
    "pretrained_model_path": null,
    "pred_save_path": null
}'

# Create the config file with the JSON content
echo "$JSON_CONTENT" > "$CONFIG_FILE"

# Provide feedback to the user
echo "Configuration file '$CONFIG_FILE' created successfully!"

make

make run

make clean

# Output in terminal is
# Epoch: 0
#    Batch: 0 / 9
#    Batch: 1 / 9
#    Batch: 2 / 9
#    Batch: 3 / 9
#    Batch: 4 / 9
#    Batch: 5 / 9
#    Batch: 6 / 9
#    Batch: 7 / 9
#    Batch: 8 / 9
#    Batch: 9 / 9
#    MSE: 0.213384 
