import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model
import numpy as np
from regularization import *

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
        accuracy_metric = tf.keras.metrics.Accuracy()

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
            "accuracy": accuracy * 100
        }



