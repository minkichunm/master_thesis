import tensorflow as tf
import os
from init_neuralnetwork import CompressibleNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from utils import *
import pickle
import json
from Compressible_Huffman import Huffman
import tensorflow.compat.v1 as tf1

regularization_loss_results = []
ce_loss_results = []
accuracy_results = []

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        regularization_loss_results.append(logs['regularization_loss'])
        ce_loss_results.append(logs['loss_cross_entropy'])
        accuracy_results.append(logs['accuracy'])  

def compress_NN_param(options, x_train, y_train, x_test, y_test, train_generator, steps_per_epoch):
    if options["load_model"]: # load the saved weights 
        print("Start compression with saved weights")
	    
    else:
        print("Start compression")
    dataset_classes = {
    "mnist": 10,
    "cifar": 10,
    "celeba": 2,
    "3d": 1
    }	
    # set parameters for training
    coefficients = options["coefficients"]
    num_epoch = options["epoch"]
    batch_size = options["batch_size"]
    regularization_type = options["regularization_type"]
    directory = "results/" + options["directory_path"]
    scale_outlier = options["scale_outlier"]
    loss_results = []
    num_class = dataset_classes[options["dataset"]]	
    selected_model = options["model"]
    val_accuracy_results = []
    file_size_results = []
    val_loss_results = []
    compressibleNN_list = []
  
    # Check if the directory exists and create it if necessary
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # select model
    selected_model = load_model_function(options["model"])
    
    # Create a list of models by copying the base model
    for coeff in coefficients:
        model_instance = CompressibleNN(selected_model(input_shape = x_train.shape[1:], num_classes = num_class), coeff, regularization_type, scale_outlier)
        compressibleNN_list.append(model_instance)
    
    # Train the model with the current hyperparameters
    #optimizer = tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=.95)
    #optimizer = tf.optimizers.Adadelta(rho=0.9, epsilon=1e-7)
    #optimizer = tf.optimizers.Adagard(learning_rate=1e-3, epsilon=1e-7)
    optimizer = tf.optimizers.SGD(learning_rate=0.01)
    
    # Get the optimizer configuration as a dictionary
    optimizer_config = optimizer.get_config()

    # Convert the dictionary to a formatted string
    optimizer_info = ", ".join(f"{key}={value}" for key, value in optimizer_config.items())

    # Print the optimizer information in one line
    print(f"Optimizer: {optimizer_info}")
    
    for count, compressibleNN in enumerate(compressibleNN_list):
        # Create a learning rate scheduler
        #lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
        compressibleNN.compile(optimizer,loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

        if options["load_model"]: # load the saved weights 
            compressibleNN.load_weights(f'{directory}/model_{count}')
        
        checkpoint_callback = ModelCheckpoint(f'{directory}/model_{count}',
                                      save_weights_only=True, mode='max',
                                      monitor="val_accuracy",
                                      save_best_only=True)
             
        history = compressibleNN.fit(train_generator, epochs=num_epoch, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                                     validation_data=(x_test, y_test), callbacks=[CustomCallback(), checkpoint_callback])
        
        celoss = history.history['loss_cross_entropy'][0]
        regloss = history.history['regularization_loss'][0]
    
        val_accuracy_results.append(history.history['val_accuracy'])
        val_loss_results.append(history.history['val_loss'])
    
        print(f"{compressibleNN.reg_type} {compressibleNN.regularization_coefficient} done")
        #print(history.history['val_accuracy'])
        #print(history.history.keys())
        
    # Define the full path to the log file
    reg_filename = os.path.join(directory, "loss_logs.txt")
    ce_filename = os.path.join(directory, "ce_logs.txt")
    options_filename = os.path.join(directory, "options_logs.txt")
    accuracy_filename = os.path.join(directory, "accuracy_logs.txt")
    weights_size_filename = os.path.join(directory, "weights_size.txt")
    val_accuracy_filename = os.path.join(directory, "val_accuracy_logs.txt")
    val_loss_filename = os.path.join(directory, "val_loss_logs.txt")

    # Save the original weights
    for count, compressibleNN in enumerate(compressibleNN_list):
        original_weights = compressibleNN.net_model.get_weights()
        weights_filename = f'{directory}/original_model{count}_weights.pkl'

        with open(weights_filename, 'wb') as file:
            pickle.dump(original_weights, file)
            

    # Convert the TensorFlow tensors to normal Python floats and round to 3 decimal places
    precision = 3
    ce_loss_result = [round(float(ce_loss), precision) for ce_loss in ce_loss_results]
    regularization_loss_result = [round(float(regularization_loss), precision) for regularization_loss in regularization_loss_results]
    accuracy_result = [round(float(acc), precision) for acc in accuracy_results]
    val_loss_result = [round(float(acc), precision) for sublist in val_loss_results for acc in sublist]
    val_accuracy_result = [round(float(acc) * 100, precision) for sublist in val_accuracy_results for acc in sublist]

    # Write about options
    options_str = json.dumps(options)  # Convert the options dictionary to a JSON string
    options_filename = os.path.join(directory, "options_logs.txt")
    with open(options_filename, "a") as file:
        file.write(options_str + "\n")
        file.write(json.dumps(optimizer_config) + "\n")  # Convert and write optimizer_config as JSON

    step = len(ce_loss_result) // len(coefficients)

    # Write data about accuracy, validation, validation loss
    write_data_to_file(os.path.join(directory, "accuracy_logs.txt"), accuracy_result, step)
    write_data_to_file(os.path.join(directory, "val_accuracy_logs.txt"), val_accuracy_result, step)
    write_data_to_file(os.path.join(directory, "val_loss_logs.txt"), val_loss_result, step)
    write_data_to_file(os.path.join(directory, "loss_logs.txt"), regularization_loss_result, step)
    write_data_to_file(os.path.join(directory, "ce_logs.txt"), ce_loss_result, step)

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

    write_to_file(os.path.join(directory, "weights_size.txt"), file_size_results)

    print("Compression done")
    
    
