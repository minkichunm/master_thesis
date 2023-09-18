import tensorflow as tf
from tensorflow import keras

class RegularizationCallback(keras.callbacks.Callback):
    def __init__(self, regularization_weighting_coefficient, regularization_type):
        super(RegularizationCallback, self).__init__()
        self.regularization_weighting_coefficient = regularization_weighting_coefficient
        self.regularization_type = regularization_type
        
    def on_epoch_begin(self, epoch, logs=None):
                
    def on_train_batch_begin(self, batch, logs=None):
        pass  # Add custom batch logic here if needed
    
    def on_train_batch_end(self, batch, logs=None):
        pass  # Add custom batch logic here if needed

    def on_epoch_end(self, epoch, logs=None):
        pass  # Add custom epoch-end logic here if needed

    def on_train_begin(self, batch, logs=None):
        if self.regularization_type == 'entropy':
            self.model.train_step_type = 'entropy'
        elif self.regularization_type == 'sparsity':
            self.model.train_step_type = 'sparsity'
        else:
            raise ValueError("Invalid regularization_type. Use 'entropy' or 'sparsity'.")
    
    def on_batch_end(self, batch, logs=None):
        self.model.train_step_type = None

        
        
#         class RegularizationCallback(keras.callbacks.Callback):
#     def __init__(self, coefficients, regularization_type, update_schedule):
#         super(RegularizationCallback, self).__init__()
#         self.coefficients = coefficients
#         self.regularization_type = regularization_type
#         self.update_schedule = update_schedule  # List of tuples (epoch, coefficient)
        
#     def on_epoch_begin(self, epoch, logs=None):
#         for schedule_epoch, coefficient in self.update_schedule:
#             if epoch == schedule_epoch:
#                 self.model.regularization_weighting_coefficient.assign(coefficient)
#                 print(f"Updated regularization_weighting_coefficient to {coefficient} at epoch {epoch}")
                
#     def on_train_batch_begin(self, batch, logs=None):
#         pass  # Add custom batch logic here if needed
    
#     def on_train_batch_end(self, batch, logs=None):
#         pass  # Add custom batch logic here if needed

#     def on_epoch_end(self, epoch, logs=None):
#         pass  # Add custom epoch-end logic here if needed

#     def on_batch_begin(self, batch, logs=None):
#         if self.regularization_type == 'entropy':
#             self.model.train_step_type = 'entropy'
#         elif self.regularization_type == 'sparsity':
#             self.model.train_step_type = 'sparsity'
#         else:
#             raise ValueError("Invalid regularization_type. Use 'entropy' or 'sparsity'.")
    
#     def on_batch_end(self, batch, logs=None):
#         self.model.train_step_type = None

# regularization_callback = RegularizationCallback(coefficients, regularization_type, update_schedule)
