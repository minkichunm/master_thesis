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
from utils import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def __create_options():
    options = {
        "directory_path": "results/temp",
        "dataset": "cifar",
    }

    DEBUG_MODE = 0

    p = argparse.ArgumentParser(description='Decompressing neural network parameter')
    p.add_argument('-dir', '--directory_path', type=str,
                   help='Path to the output directory (default: "results_")')
    p.add_argument('-ds', '--dataset', type=str,
                   help='Choose dataset "mnist", "cifar","celeba","3d" (default: "cifar")')
               
    p.add_argument('-debug', '--debug', action='store_true',
                   help='Enable DEBUG MODE for debugging output')
               
    args, _ = p.parse_known_args()  # Use parse_known_args to ignore extra arguments
    
    if args.directory_path:
        options["directory_path"] = args.directory_path
    if args.dataset:
        options["dataset"] = args.dataset
        
    if args.debug:
        print(options)

    return options
    

options = __create_options()

print("Decompression start")

directory = "results/" + options["directory_path"]
dataset = options["dataset"]
compNN_list = []
cnt = 0
decompressed_weights_list = []


# Load the CompressibleNN instance
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

    #print(f"Differences between original and decompressed weights in model{cnt}:")
    #for i, diff in enumerate(differences):
    #    print(f"Layer {i+1}: Max Difference = {np.max(diff)}, Mean Difference = {np.mean(diff)}")
    #print("-"*80)

# set decompressed weights
for compNN, decompressed_weights in zip(compNN_list, decompressed_weights_list):
    compNN.set_weights(decompressed_weights)

# Load the dataset
if dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif dataset == "cifar":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
elif dataset == "celeba":
    celeba_folder = 'celeba_dataset/'
    train_samples = 100
    validation_samples = 25
    test_samples = 100
    height = 218 
    width = 178
    input_shape = (height, width, 3)
    df = pd.read_csv(celeba_folder+'list_attr_celeba.csv', index_col=0)
    df_partition_data = pd.read_csv(celeba_folder+'list_eval_partition.csv')
    df_partition_data.set_index('image_id', inplace=True)
    df = df_partition_data.join(df['Male'], how='inner')
    x_train, y_train = generate_df(0, 'Male', train_samples, df, celeba_folder)
    datagen_train =  ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    x_test, y_test = generate_df(1, 'Male', validation_samples, df, celeba_folder)
    # Preparing train data with data augmentation
    datagen_val =  ImageDataGenerator(
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
    )
        
# Reshape y_test to match the shape of predicted_labels using TensorFlow
y_test_flat = tf.cast(tf.squeeze(y_test), tf.int64)

acc_results = []

for cnt, compNN in enumerate(compNN_list): 
    # Make predictions using your model
    predictions = compNN.predict(x_test)
    # Calculate top-1 accuracy
    
    predicted_labels = tf.argmax(predictions, axis=1)

    top_1_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, y_test_flat), tf.float32)) * 100

    print(f"Model{cnt} Top-1 Accuracy: {top_1_accuracy:.2f}%")
    acc_results.append(f"{top_1_accuracy:.2f}")

print("Save accuracy log file")
# Define the full path to the log file
log_filename = os.path.join(directory, "accuracy.txt")

# Save the results to the file
write_to_file(log_filename, acc_results)
        
print("Decompression done")

