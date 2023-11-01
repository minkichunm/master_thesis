import tensorflow as tf
from tensorflow import keras
import pickle
from Compressible_Huffman import Huffman
import os
from utils import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def decompress_NN_param(options, x_train, y_train, x_test, y_test, train_generator, steps_per_epoch):
    print("Start decompression")

    # Init
    directory = "results/" + options["directory_path"]
    cnt = 0
    compNN_list = []
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

        # Print or log differences as needed
        # print(f"Differences between original and decompressed weights in model{cnt}:")
        # for i, diff in enumerate(differences):
        #     print(f"Layer {i + 1}: Max Difference = {np.max(diff)}, Mean Difference = {np.mean(diff)}")
        # print("-" * 80)

    # Set decompressed weights
    for compNN, decompressed_weights in zip(compNN_list, decompressed_weights_list):
        compNN.set_weights(decompressed_weights)

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
