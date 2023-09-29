from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import dahuffman
import pickle
from Compressible_Huffman import Huffman
from tensorflow.keras.datasets import mnist
import os
import argparse
import sys

def __create_options():
    options = {
        "directory_path": "results_",
    }

    DEBUG_MODE = 0

    p = argparse.ArgumentParser(description='Decompressing neural network parameter')
    p.add_argument('-d', '--directory_path', type=str,
                   help='Path to the output directory (default: "results_")')
               
    p.add_argument('-debug', '--debug', action='store_true',
                   help='Enable DEBUG MODE for debugging output')
               
    args, _ = p.parse_known_args()  # Use parse_known_args to ignore extra arguments
    
    if args.directory_path:
        options["directory_path"] = args.directory_path
        
    if args.debug:
        print(options)

    return options
    

options = __create_options()

print("Decompression start")


# Load the CompressibleNN instance

directory = options["directory_path"]
compNN_list = []
cnt = 0
decompressed_weights_list = []

while True:
    # Check if the file exists
    filename = f'{directory}/compressed_nn_{cnt}.pkl'
    if not os.path.exists(filename):
        break

    # Load the compressed model from the file
    with open(filename, 'rb') as compressed_nn:
        model_architecture = pickle.load(compressed_nn)
        net_model = tf.keras.models.model_from_json(model_architecture)
        codec = pickle.load(compressed_nn)
        compressed_model_weights = pickle.load(compressed_nn)

    # Create a new instance of CompressibleNN
    compNN = Huffman(net_model)
    compNN.codec = codec
    compNN_list.append(compNN)
    decompressed_weights = compNN.decompressNN(compressed_model_weights)
    decompressed_weights_list.append(decompressed_weights)
    
    # Increment the counter
    cnt += 1

# Compare the original weights with the decompressed weights
original_model_weights_list = []
for cnt, compNN in enumerate(compNN_list):
    # Check if the file exists
    filename = f'{directory}/original_model{cnt}_weights.pkl'
    if not os.path.exists(filename):
        break
    with open(filename, 'rb') as file:
        original_model_weights = pickle.load(file)
    original_model_weights_list.append(original_model_weights)
        
    differences = compNN.compare_weights(original_model_weights_list[cnt], decompressed_weights_list[cnt])

    print(f"Differences between original and decompressed weights in model{cnt}:")
    for i, diff in enumerate(differences):
        print(f"Layer {i+1}: Max Difference = {np.max(diff)}, Mean Difference = {np.mean(diff)}")
    print("-"*80)

# set decompressed weights
for compNN, decompressed_weights in zip(compNN_list, decompressed_weights_list):
    compNN.set_weights(decompressed_weights)


# Load the CIFAR dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1] # 32
input_size = image_size * image_size

# Preprocess the data (reshape and normalize)
#x_test = np.reshape(x_test, [-1, image_size, image_size, 1])  # mnist
x_test = np.reshape(x_test, [-1, image_size, image_size, 3])  # cifar Keep the shape (None, 32, 32, 3)

# Reshape y_test to match the shape of predicted_labels using TensorFlow
y_test_flat = tf.cast(tf.squeeze(y_test), tf.int64)

results = []

for cnt, compNN in enumerate(compNN_list): 
    # Make predictions using your model
    predictions = compNN.predict(x_test)
    # Calculate top-1 accuracy
    
    predicted_labels = tf.argmax(predictions, axis=1)

    top_1_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, y_test_flat), tf.float32)) * 100

    print(f"Model{cnt} Top-1 Accuracy: {top_1_accuracy:.3f}%")
    results.append(f"Model{cnt} Top-1 Accuracy: {top_1_accuracy:.3f}%")

print("Save accuracy log file")
# Define the full path to the log file
log_filename = os.path.join(directory, "accuracy_logs.txt")

# Save the results to the file
with open(log_filename, "a") as file:
    for result_entry in results:
        file.write(result_entry + "\n")
        
print("Decompression done")

