import tensorflow as tf
import numpy as np
import dahuffman
import pickle

from tensorflow import keras

# new task 
# one main jupyter notebook entroy -> saves model -> load model

# two jupyter notebook
# one of them saves model with two files weights, codec
# one of them loads model
# save an object itself ( everything )

# x axis model size
# y accuracy
# fig 7
# ramda if too high too compress

# CNN
# benchmark

class CompressibleNN(keras.Model):
    def __init__(self, net_model):
        super(CompressibleNN, self).__init__()
        self.net_model = net_model
        self.codec = []
#         self.CompressibleNN

    def compressNN(self):
        # Reshape the weights
        reshaped_weights = self.reshape_weights()
        
        # Compress the weights using Huffman coding
        compressed_weights = []
        for i, weight_tensor in enumerate(reshaped_weights):
            weight_flattened = weight_tensor.flatten()
            weight_bytes = weight_flattened.tobytes()  # Convert to byte array
            encoder = dahuffman.HuffmanCodec.from_data(weight_bytes)
            self.codec.append(encoder) 
            compressed_data = encoder.encode(weight_bytes)  # Use encode() with byte array
            compressed_weights.append(compressed_data)
        
        # Save the weights to a file
        with open('compressed_model_weights.pkl', 'wb') as file:
            pickle.dump(compressed_weights, file)
        
        return compressed_weights


    def decompressNN(self, compressed_weights):
        # Decompress the weights using Huffman coding
        decompressed_weights = []
        for i, compressed_data in enumerate(compressed_weights):
            decoder = self.codec[i]
            decompressed_data = decoder.decode(compressed_data)  # Use decode() directly

            weight_shape = self.net_model.get_weights()[i].shape  # Retrieve the shape of the corresponding weight tensor
            
            decompressed_array = np.frombuffer(bytes(decompressed_data), dtype=np.float32)
            decompressed_weights.append(decompressed_array.reshape(weight_shape))

        return decompressed_weights

    def call(self, inputs):
        return self.net_model(inputs)

    def reshape_weights(self):
        reshaped_weights = []
        for weight_tensor in self.net_model.get_weights():
            weight_shape = weight_tensor.shape
            weight_size = np.prod(weight_shape)
            # Reshape the weight tensor based on the size of the input array
            reshaped_weights.append(weight_tensor.reshape(weight_shape))
        return reshaped_weights
    
    def compare_weights(self, original_weights, decompressed_weights):
        differences = []
        for orig, decomp in zip(original_weights, decompressed_weights):
            orig_shape = orig.shape
            decomp_shape = decomp.shape
            decomp_reshaped = decomp.reshape(orig_shape) if orig_shape == decomp_shape else decomp
            diff = np.abs(orig - decomp_reshaped)
            differences.append(diff)
        return differences
