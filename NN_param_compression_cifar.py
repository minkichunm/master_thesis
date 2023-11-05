import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model, load_model
import numpy as np
import dahuffman
import pickle
from Compressible_Huffman import Huffman
from regularization import *
from getModel import *
from utils import *
import os
import sys
import json
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("compression start")
# TensorFlow example
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def __create_options():
    options = {
        "directory_path": "results_",
        "epoch": 10,
        "model": "1",
        "regularization_type" : "entropy",
        "coefficients": [1.0],
        "batch_size": 64,
        "scale_outlier": 3,
        "dataset": "cifar",
        "load_model": False,        
    }

    p = argparse.ArgumentParser(description='Compressing neural network parameter')
    p.add_argument('-dir', '--directory_path', type=str,
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
    p.add_argument('-so', '--scale_outlier', type=int,
                   help='Standard deviation for outlier setting (default: 3)')
    p.add_argument('-ds', '--dataset', type=str,
                   help='Choose dataset "mnist", "cifar","celeba","3d" (default: "cifar")')
    p.add_argument('-L', '--load_model', action='store_true', help='Load a saved model')
              

    args = p.parse_args()

    if args.directory_path:
        options["directory_path"] = args.directory_path
    if args.epoch:
        options["epoch"] = args.epoch
    if args.model:
        options["model"] = args.model
    if args.regularization_type:
        options["regularization_type"] = args.regularization_type
    if args.coefficients:
        options["coefficients"] = args.coefficients
    if args.batch_size:
        options["batch_size"] = args.batch_size
    if args.scale_outlier:
        options["scale_outlier"] = args.scale_outlier
    if args.dataset:
        options["dataset"] = args.dataset
    if args.load_model:
        options["load_model"] = args.load_model
        
    print(options)

    return options
    
options = __create_options()

# set parameters for training
coefficients = options["coefficients"]
num_epoch = options["epoch"]
batch_size = options["batch_size"]
regularization_type = options["regularization_type"]
directory = "results/" + options["directory_path"]
scale_outlier = options["scale_outlier"]
dataset = options["dataset"]
loss_results = []
num_class = 10 # mnist or cifar

# Load data set "mnist", "cifar","celeba","3d"
if dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

elif dataset == "cifar":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_generator = data_generator.flow(x_train, y_train, batch_size)
    steps_per_epoch = x_train.shape[0] // batch_size
	
elif dataset == "celeba":
    celeba_folder = 'celeba_dataset/'
    train_samples = 3000
    validation_samples = 2000
    test_samples = 1000
    height = 218 
    width = 178
    input_shape = (height, width, 3)
    df = pd.read_csv(celeba_folder+'list_attr_celeba.csv', index_col=0)
    df_partition_data = pd.read_csv(celeba_folder+'list_eval_partition.csv')
    df_partition_data.set_index('image_id', inplace=True)
    df = df_partition_data.join(df['Male'], how='inner')
    x_train, y_train = generate_df(0, 'Male', train_samples, df, celeba_folder)
    train_generator =  ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

#    datagen_train.fit(x_train)

#    train_generator = datagen_train.flow(
#      x_train, y_train,
#      batch_size=batch_size,
#    )

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
    num_class = 1

#y_test_flat = tf.cast(tf.squeeze(y_test), tf.int64)

#y_test_one_hot = tf.one_hot(y_test, depth=10)

#    datagen_val.fit(x_test)

#    val_generator = datagen_val.flow(
#        x_test, y_test,
#        batch_size=batch_size,
#    )
    
#    y_train = tf.argmax(y_train, axis=1)
#    y_test = tf.argmax(y_test, axis=1)
    
# generate a NN model
model_functions = {
    1: get_mobilenetv2,
    2: get_resnet50,
    3: get_32model,
    4: get_simplemodel,
    5: get_modified_model,
    6: get_modified_model2,
    7: get_modified_cifar10_model,
    8: get_modified_cifar10_model2,
    9: get_modified_cifar10_model3,
    10: get_celeba_temp,
    11: get_celeba_temp2,
    12: get_modified_model3,
    13: get_modified_cifar10_model4,
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

# show graph that is compressed
#variables=model.layers[1].variables[0]
#visualize_histogram(variables)

accuracy_metric = tf.keras.metrics.Accuracy()

class CompressibleNN(keras.Model):
    def __init__(self, net_model, coeff, reg_type, scale_outlier):
        super(CompressibleNN, self).__init__()

        # Build the model architecture
        self.net_model = net_model
        self.regularization_coefficient = coeff
        self.reg_type = reg_type
        #self.ce = keras.losses.categorical_crossentropy()
        self.ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.scale_outlier = scale_outlier

    def call(self, inputs):
        return self.net_model(inputs)
    
    def get_entropy_loss(self, inputs):
        entropy = 0
        
        for l in self.net_model.layers:
            if isinstance(l, keras.layers.Dense):
                for v in l.trainable_variables:
                    v_entropy, v_range = calculate_entropy(v, self.scale_outlier)
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
    
    @tf.function
    def train_step(self, input):
        images = input[0]
        labels = input[1] 
        
        with tf.GradientTape() as tape:
            output = self.net_model(images)
            val_output = self.net_model(x_test)
            
            if self.reg_type == 'entropy':
                regularization_loss = self.get_entropy_loss(images)
            elif self.reg_type == 'sparsity':
                regularization_loss = self.get_sparsity_loss(images)
            else:
                raise ValueError("Invalid reg_type. Use 'entropy' or 'sparsity'.")
                
            # compute the full loss including the cross entropy for classification and regularization for compression
            loss_cross_entropy= self.ce(labels, output)
            if self.regularization_coefficient > 0:
                loss = loss_cross_entropy + self.regularization_coefficient * regularization_loss
            else:
                loss = loss_cross_entropy
            
            # Calculate accuracy
            accuracy_metric.reset_states()
            accuracy_metric.update_state(labels, tf.argmax(output, axis=1))  # Assuming it's a classification task
            accuracy = accuracy_metric.result()
            
        # Get the gradients w.r.t the loss
        gradient = tape.gradient(loss, self.net_model.trainable_variables)
        # Update the weights using the generator optimizer
        self.optimizer.apply_gradients(
            zip(gradient, self.net_model.trainable_variables)
        )
        
        return {
            "loss_cross_entropy": loss_cross_entropy,
            "regularization_loss": regularization_loss,
            "accuracy": accuracy * 100,
#            "val_accuracy": val_accuracy * 100
        }

# Create a list of models by copying the base model
compressibleNN_list = []

for coeff in coefficients:
	model_instance = CompressibleNN(model_function(input_shape = x_train.shape[1:], num_classes = num_class), coeff, regularization_type, scale_outlier)
	compressibleNN_list.append(model_instance)


# Check if the directory exists and create it if necessary
if not os.path.exists(directory):
    os.makedirs(directory)

regularization_loss_results = []
ce_loss_results = []
tot_loss_results = []
accuracy_results = []
val_accuracy_results = []
file_size_results = []
val_loss_results = []

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        regularization_loss_results.append(logs['regularization_loss'])
        ce_loss_results.append(logs['loss_cross_entropy'])
        accuracy_results.append(logs['accuracy'])  

if options["load_model"]:
	print("Training with pretrained")        
        
for count, compressibleNN in enumerate(compressibleNN_list):
    optimizer = tf.optimizers.Adam(learning_rate=1e-3, beta_1=0.9)
    compressibleNN.compile(optimizer,loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    if options["load_model"]: # load the saved model here
        compressibleNN.load_weights('{directory}/model_{count}')
        history = compressibleNN.fit(train_generator, epochs=num_epoch, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                  validation_data=(x_test, y_test), callbacks=[CustomCallback()])
    else:
        history = compressibleNN.fit(train_generator, epochs=num_epoch, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
              validation_data=(x_test, y_test), callbacks=[CustomCallback()])
    # Train the model with the current hyperparameters
    #    history = compressibleNN.fit(x=x_train, y=y_train, epochs=num_epoch, batch_size=batch_size,
    #          validation_data=(x_test, y_test), callbacks=[CustomCallback()])
    compressibleNN.save_weights(f'{directory}/model_{count}')
    
#    history = compressibleNN.fit(train_generator, epochs=num_epoch, batch_size=batch_size,#steps_per_epoch=len(x_train) // batch_size, validation_steps=len(x_test) // batch_size
#    validation_data=val_generator, callbacks=[CustomCallback()])          
    
    celoss = history.history['loss_cross_entropy'][0]
    regloss = history.history['regularization_loss'][0]
    
    val_accuracy_results.append(history.history['val_accuracy'])
    val_loss_results.append(history.history['val_loss'])
    
    print(f"{compressibleNN.reg_type} {compressibleNN.regularization_coefficient} done")
    print(history.history.keys())
    #print(history.history['val_loss'])
    #print(val_loss_results)
    
# Save the original weights
# Define the full path to the log file
reg_filename = os.path.join(directory, "loss_logs.txt")
ce_filename = os.path.join(directory, "ce_logs.txt")
options_filename = os.path.join(directory, "options_logs.txt")
accuracy_filename = os.path.join(directory, "accuracy_logs.txt")
weights_size_filename = os.path.join(directory, "weights_size.txt")
val_accuracy_filename = os.path.join(directory, "val_accuracy_logs.txt")
val_loss_filename = os.path.join(directory, "val_loss_logs.txt")

for count, compressibleNN in enumerate(compressibleNN_list):
    original_weights = compressibleNN.net_model.get_weights()
    weights_filename = f'{directory}/original_model{count}_weights.pkl'
    
    with open(weights_filename, 'wb') as file:
        pickle.dump(original_weights, file)


# Convert the TensorFlow tensors to normal Python floats and round to 3 decimal places
precision = 3
regularization_loss_result = [round(float(regularization_loss), precision) for regularization_loss in regularization_loss_results]
ce_loss_result = [round(float(ce_loss), precision) for ce_loss in ce_loss_results]
accuracy_result = [round(float(acc), precision) for acc in accuracy_results]
val_loss_result = [round(float(acc), precision) for sublist in val_loss_results for acc in sublist]
val_accuracy_result = [round(float(acc) * 100, precision) for sublist in val_accuracy_results for acc in sublist]
#print(val_accuracy_result)

# write about options
options_str = json.dumps(options)  # Convert the options dictionary to a JSON string
with open(options_filename, "a") as file:
	file.write(options_str + "\n")
	
step = len(ce_loss_result) // len(coefficients)

# write data about accuracy, validation, validation loss
write_data_to_file(accuracy_filename, accuracy_result, step)
write_data_to_file(val_accuracy_filename, val_accuracy_result, step)
write_data_to_file(val_loss_filename, val_loss_result, step)  
write_data_to_file(reg_filename, regularization_loss_result, step)
write_data_to_file(ce_filename, ce_loss_result, step)

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
    file_size_results.append(file_size_bytes)

write_to_file(weights_size_filename, file_size_results)

print("compression done")

