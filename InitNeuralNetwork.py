import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model
import numpy as np

class initNeuralNetwork(keras.Model):
    def __init__(self, net_model):
        super(initNeuralNetwork, self).__init__()
        self.net_model = net_model
        self.CE_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, inputs):
        return self.net_model(inputs)
    
    def entropy_loss(self, inputs):
        entropy = 0
        
        for l in self.net_model.layers:
            if isinstance(l, keras.layers.Dense):
                for v in l.trainable_variables:
                    v_entropy, v_range = calculate_entropy(v)
                    entropy += v_entropy
        
        return entropy
    
    def regularization_loss(self, inputs, regularization_weighting_coefficient = 1e-2):
        rg_loss = 0
        num_activations = 0  # Initialize the total number of activations
        
        for l in self.net_model.layers:
            if isinstance(l, keras.layers.Dense):
                for v in l.trainable_variables:
                    num_activations += tf.reduce_prod(v.shape)  # Accumulate the number of activations
                    v_regularization_loss = calc_sparsity_regularization(v)
                    rg_loss += v_regularization_loss

        # Divide the total regularization loss by the total number of activations
        rg_loss = rg_loss * regularization_weighting_coefficient / tf.cast(num_activations, dtype=tf.float32)
        
        return rg_loss
    
    
    def train_step(self, input):
        images = input[0]
        labels = input[1]

        with tf.GradientTape() as tape:
            output = self.net_model(images)
            loss =  self.entropy_loss(images)
            regularization_loss = self.regularization_loss(images)

        # 1. Entropy-based regularization
        # Get the gradients w.r.t the loss
        gradient = tape.gradient(loss, self.net_model.trainable_variables)
        # Update the weights using the generator optimizer
        self.optimizer.apply_gradients(
            zip(gradient, self.net_model.trainable_variables)
        )
        return {"loss": loss}

        # 2. Sparsity-based regularization
#         gradient_reg = tape.gradient(regularization_loss, self.net_model.trainable_variables)
#         # Update the weights using the generator optimizer
#         self.optimizer.apply_gradients(
#             zip(gradient_reg, self.net_model.trainable_variables)
#         )
#         return {"regularization loss": regularization_loss}
    