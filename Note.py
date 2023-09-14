# HPC

# start jupyter notebook
# ~/.local/bin/jupyter-notebook

# NN param
# #  Steps of the work:

# Make a model with accessible parameters using TensorFlow 2.x
# Write an entropy loss and add it to the network parameters for training.
# Train a model by jointly optimizing its loss (e.g. cross entropy for classification) and the entropy for compressability.
# Compress the model parameters using Huffman coding.
# Analyze the results and trade-off between accuracy and compressability.
# Write a model wrapper that can read and write the compressed parameters.

# Notes:

# Can be started on simple datasets like MNIST.
# Can start from simple MLP NNs. And then extend to convs. The implementation should be generic enough so that it can support any layer.
# Bonus: add support for normalization layers: batch norm, instance norm, etc.


# sparsity regularization:
# I have to  sum all of the L1 and then divides by the total number of activations.
# but I currently have this number of activations that is inside the summation, and for every layer I do a normalization on my own. 
# I have to do this in the regularization loss I have after you sum all the values. I have to also sum all the activations and then divide

# hauffman coding
# save only we needed
# small size as possible
# save model 
# see if size of model
# save a lot of time
# only model
# only weights
# check if they are similar
# if it does 
# if not set weight 0
# check if pickle does already does huffman
# accuracy with changing ramda
# direct work on test
# how do I prove this thing works
# test function ? change class? that format? that type of NN
# force to u test 
# make sure fix discussed

# one file initiate 
# load model test testset
# vary lamda start with lamda 0
# if pandas uses huffman?

# large lamda 
# train model with 
# original training model
# effect of using lamda

# send 2 files

# changed calculate_histogram_range

However, pickle itself does not perform data compression like Huffman coding.
so i need to set all its weights to 0 for example

You make these two files I want to have one file that you first initiate the model. You train the model, you save the model in the other file. I want you to load the model and test it on your test set.
The size of the file generated by pickling a neural network model doesn't depend on whether the weights are all set to 0 or not

During training, especially if you use certain callbacks or custom code within your model, the model might accumulate references to objects or resources that cannot be pickle


ValueError: Model <__main__.CompressibleNN object at 0xffff1a3d5570> cannot be saved either because the input shape is not available or because the forward pass of the model is not defined.To define a forward pass, please override `Model.call()`. To specify an input shape, either call `build(input_shape)` directly, or call the model on actual data using `Model()`, `Model.fit()`, or `Model.predict()`. If you have a custom training step, please make sure to invoke the forward pass in train step through `Model.__call__`, i.e. `model(inputs)`, as opposed to `model.call()`.


