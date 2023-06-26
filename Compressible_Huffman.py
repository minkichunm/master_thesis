import tensorflow as tf
import numpy as np
import dahuffman
import pickle

from tensorflow import keras

class CompressibleNN(keras.Model):
    def __init__(self, net_model):
        super(CompressibleNN, self).__init__()
        self.net_model = net_model
        self.codec = []

    def compressNN(self, inputs):
        # Reshape the weights
        reshaped_weights = self.reshape_weights(inputs)
#         print(len(reshaped_weights))

        # Compress the weights using Huffman coding
        compressed_weights = []
        for i, weight_tensor in enumerate(reshaped_weights):
            weight_flattened = weight_tensor.flatten()
            weight_bytes = weight_flattened.tobytes()  # Convert to byte array
            encoder = dahuffman.HuffmanCodec.from_data(weight_bytes)
            self.codec.append(encoder) 
            compressed_data = encoder.encode(weight_bytes)  # Use encode() with byte array
            compressed_weights.append(compressed_data)
            print("Weight tensor shape:", weight_tensor.shape)
            print("Weight tensor size:", weight_tensor.size)
            print("Weight flattened size:", weight_flattened.size)
            print("Compressed data size:", len(compressed_data))
        
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

            print("Decompressed data size:", len(decompressed_data))
            weight_shape = self.net_model.get_weights()[i].shape  # Retrieve the shape of the corresponding weight tensor
            decompressed_size = np.prod(weight_shape) * np.dtype(np.float32).itemsize

            print("Expected decompressed size:", decompressed_size)
            print("Actual decompressed size:", len(decompressed_data))

            if len(decompressed_data) < decompressed_size:
                decompressed_data += b'\x00' * (decompressed_size - len(decompressed_data))

            elif len(decompressed_data) > decompressed_size:
                decompressed_data = decompressed_data[:decompressed_size]

            decompressed_array = np.frombuffer(bytes(decompressed_data), dtype=np.float32)
            decompressed_weights.append(decompressed_array.reshape(weight_shape))

        return decompressed_weights

    def call(self, inputs):
        return self.net_model(inputs)

    def reshape_weights(self, inputs):
        reshaped_weights = []
        current_index = 0
        for weight_tensor in self.net_model.get_weights():
            weight_shape = weight_tensor.shape
            weight_size = np.prod(weight_shape)
            # Reshape the weight tensor based on the size of the input array
            reshaped_weights.append(weight_tensor.reshape(weight_shape))
            current_index += weight_size
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
