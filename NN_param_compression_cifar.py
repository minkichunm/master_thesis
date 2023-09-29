import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
import numpy as np
import dahuffman
import pickle
from Compressible_Huffman import Huffman
from regularization import *
from getModel import *
import os
import sys

print("compression start")

def __create_options():
    options = {
        "directory_path": "results_",
        "epoch": 10,
        "model": "1",
        "regularization_type" : "entropy",
        "coefficients": [1.0],
        "batch_size": 256,
    }

    DEBUG_MODE = 1

    p = argparse.ArgumentParser(description='Compressing neural network parameter')
    p.add_argument('-d', '--directory_path', type=str,
                   help='Path to the output directory (default: "results_")')
    p.add_argument('-e', '--epoch', type=int,
                   help='Number of training epochs (default: 1)')
    p.add_argument('-m', '--model', type=int,
                   help='Choose a model between [1, 2, 3, 4] (default: 1)\n'
                        '1: get_model()\n'
                        '2: get_3_model()\n'
                        '3: get_32_model()\n'
                        '4: get_simple_model()')
    p.add_argument('-type', '--regularization_type', type=str,
                   help='Choose regularization type between "sparsity" and "entropy" (default: "entropy")')
    p.add_argument('-coeff', '--coefficients', type=float, nargs='*', default=[1.0],
                   help='Coefficients for a custom option (default: [1.0])')
    p.add_argument('-b', '--batch_size', type=int,
                   help='Batch size for training (default: 256)')
               
    p.add_argument('-debug', '--debug', action='store_false',
                   help='Unenable DEBUG MODE for debugging output')
               

    args = p.parse_args()

    if args.directory_path:
        options["directory_path"] = args.directory_path
    if args.epoch:
        options["epoch"] = args.epoch
    if args.model:
        options["model"] = args.model
    if args.regularization_type:
        options["type"] = args.regularization_type
    if args.coefficients:
        options["coefficients"] = args.coefficients
    if args.batch_size:
        options["batch_size"] = args.batch_size
        
    if args.debug:
        print(options)

    return options
    

options = __create_options()

# Load Cifar data set
train_set, test_set = tf.keras.datasets.cifar10.load_data()
#train_set, test_set = mnist.load_data()


# set parameters for training
coefficients = options["coefficients"]
num_epoch = options["epoch"]
batch_size = options["batch_size"]
regularization_type = options["type"]
directory = options["directory_path"]
results = []
loss_results = []

# generate a NN model
model_functions = {
    1: get_model,
    2: get_3model,
    3: get_32model,
    4: get_simplemodel
}


# Assuming options["model"] contains the selected model value
selected_model = options["model"]

# Check if the selected model is in the dictionary, and call the corresponding function
try:
    if selected_model in model_functions:
        model_function = model_functions[selected_model]
    else:
        raise Exception(f"Error: Invalid model choice '{selected_model}'. Please choose a valid model.")
except Exception as e:
    print(str(e))
    sys.exit(-1)

model = model_function()

# show graph that is compressed
#variables=model.layers[1].variables[0]
#visualize_histogram(variables)

class CompressibleNN(keras.Model):
    def __init__(self, net_model, coeff, reg_type):
        super(CompressibleNN, self).__init__()

        # Build the model architecture
        self.net_model = net_model
        self.regularization_coefficient = coeff
        self.reg_type = reg_type
        self.ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, inputs):
        return self.net_model(inputs)
    
    def get_entropy_loss(self, inputs):
        entropy = 0
        
        for l in self.net_model.layers:
            if isinstance(l, keras.layers.Dense):
                for v in l.trainable_variables:
                    v_entropy, v_range = calculate_entropy(v)
                    entropy += v_entropy
        entropy = entropy / tf.experimental.numpy.log2(tf.cast(nbins, tf.float32))
        return entropy
    
    def get_sparsity_loss(self, inputs):
        rg_loss = 0
        Na = tf.cast(tf.shape(inputs)[0], dtype=tf.float32)  # Number of weights in the current training batch
        
        for l in self.net_model.layers:
            if isinstance(l, keras.layers.Dense):
                for v in l.trainable_variables:
                    v_regularization_loss = calc_sparsity_regularization(v)
                    rg_loss += v_regularization_loss
        rg_loss = rg_loss / Na
        
        return rg_loss
    
    def train_step(self, input):
        images = input[0]
        labels = input[1]

        with tf.GradientTape() as tape:
            output = self.net_model(images)
            
            if self.reg_type == 'entropy':
                regularization_loss = self.get_entropy_loss(images)
            elif self.reg_type == 'sparsity':
                regularization_loss = self.get_sparsity_loss(images)
            else:
                raise ValueError("Invalid reg_type. Use 'entropy' or 'sparsity'.")
                
            # compute the full loss including the cross entropy for classification and regularization for compression
            loss_cross_entropy= self.ce(labels, output)
            if self.regularization_coefficient>0:
                loss = loss_cross_entropy + self.regularization_coefficient * regularization_loss
            else:
                loss = loss_cross_entropy
            
        # Get the gradients w.r.t the loss
        gradient = tape.gradient(loss, self.net_model.trainable_variables)
        # Update the weights using the generator optimizer
        self.optimizer.apply_gradients(
            zip(gradient, self.net_model.trainable_variables)
        )
            
        return {"loss_cross_entropy": loss_cross_entropy, "regularization_loss": regularization_loss}

# Create a list of models by copying the base model
compressibleNN_list = []

for coeff in coefficients:
	model_instance = CompressibleNN(model_function(), coeff, regularization_type)
	compressibleNN_list.append(model_instance)


# Check if the directory exists and create it if necessary
if not os.path.exists(directory):
    os.makedirs(directory)

regularization_loss_results = []
ce_loss_results = []
tot_loss_results = []

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        regularization_loss_results.append(logs['regularization_loss'])
        ce_loss_results.append(logs['loss_cross_entropy'])
        #tot_loss_results.append(logs['loss'])

for compressibleNN in compressibleNN_list:
    optimizer = tf.optimizers.Adam(learning_rate=1e-3, beta_1=0.9)
    compressibleNN.compile(optimizer, metrics=['accuracy'])

    # Train the model with the current hyperparameters
    history = compressibleNN.fit(x=train_set[0], y=train_set[1], epochs=num_epoch, batch_size=batch_size, callbacks=[CustomCallback()])
    
    #return {"loss_cross_entropy": loss_cross_entropy, "regularization_loss": regularization_loss, "loss": loss}
    #loss = history.history['loss'][0]
    celoss = history.history['loss_cross_entropy'][0]
    regloss = history.history['regularization_loss'][0]
    
    
    # Append the results
    results.append(f"{compressibleNN.reg_type}_regloss: {regloss:.2f}, loss_cross_entropy: {celoss:.2f},  coeff: {compressibleNN.regularization_coefficient}, num_epoch: {num_epoch}, batch_size: {batch_size}.")
    
    print(f"{compressibleNN.reg_type} {compressibleNN.regularization_coefficient} done")

# Save the original weights
# Define the full path to the log file
log_filename = os.path.join(directory, "loss_logs.txt")
ce_filename = os.path.join(directory, "ce_logs.txt")
#tot_filename = os.path.join(directory, "tot_loss_logs.txt")

for count, compressibleNN in enumerate(compressibleNN_list):
    original_weights = compressibleNN.net_model.get_weights()
    weights_filename = f'{directory}/original_model{count}_weights.pkl'
    
    with open(weights_filename, 'wb') as file:
        pickle.dump(original_weights, file)
        
    # Get the size of the saved file
    file_size_bytes = os.path.getsize(weights_filename)

    # Convert bytes to megabytes
    file_size_mb = file_size_bytes / (1024 * 1024)  # 1 MB = 1024 * 1024 bytes

    # Append the file size in MB to the log file
    with open(log_filename, "a") as log_file:
        log_file.write(f"Original weights: {weights_filename}, Size: {file_size_mb:.2f} MB\n")


# Convert the TensorFlow tensors to normal Python floats and round to 3 decimal places
regularization_loss_result = [round(float(regularization_loss), 3) for regularization_loss in regularization_loss_results]
ce_loss_result = [round(float(ce_loss), 3) for ce_loss in ce_loss_results]
#tot_loss_result = [round(float(tot_loss), 3) for tot_loss in tot_loss_results]

# Save the results to the log file
with open(log_filename, "a") as file:
    file.write(options + "\n")
    for result_entry in results:
        file.write(result_entry + "\n")
        
    for i in range(0, len(regularization_loss_result), num_epoch):
        epoch_results = regularization_loss_result[i:i + num_epoch]
        file.write(", ".join(map(str, epoch_results)))
    file.write("\n\n\n")
    

with open(ce_filename, "a") as file:
    for i in range(0, len(ce_loss_result), num_epoch):
        ce_loss = ce_loss_result[i:i + num_epoch]
        file.write(", ".join(map(str, ce_loss)))
    file.write("\n")

print("Save the CompressibleNN instance to a file for each model")
for count, compressibleNN in enumerate(compressibleNN_list):
    trainedNN = Huffman(compressibleNN.net_model)
    compressed_weights = trainedNN.compressNN()
    
    # Save the CompressibleNN instance to a file for each compressibleNN
    filename = f'{directory}/compressed_nn_{count}.pkl'  # Unique filename for each instance
    with open(filename, 'wb') as outp:
        pickle.dump(trainedNN.net_model.to_json(), outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trainedNN.codec, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(compressed_weights, outp, pickle.HIGHEST_PROTOCOL)
    
    # Save the CompressibleNN instance to a file for each compressibleNN
    weightfilename = f'{directory}/model{count}_compressed_weights.pkl'  # Unique filename for each instance
    with open(weightfilename, 'wb') as outp:
        pickle.dump(compressed_weights, outp, pickle.HIGHEST_PROTOCOL)
    
    # Get the size of the saved file
    file_size_bytes = os.path.getsize(weightfilename)

    # Convert bytes to megabytes
    file_size_mb = file_size_bytes / (1024 * 1024)  # 1 MB = 1024 * 1024 bytes
    
    # Append the file size in MB to the log file
    with open(log_filename, "a") as log_file:
        log_file.write(f"Compressed weights: {weightfilename}, Size: {file_size_mb:.2f} MB\n")

# Save the results to the log file
with open(log_filename, "a") as file:
    for result_entry in results:
        file.write(result_entry + "\n")

print("compression done")

