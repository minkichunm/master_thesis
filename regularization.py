import tensorflow as tf
import numpy as np
from tensorflow import keras

nbins = 256

def calculate_entropy(variables, eps=1e-10):
    min_h, max_h = calculate_histogram_range(variables)
    flat_vars = tf.reshape(variables, (-1,1))    
    hist = calculate_histogram(flat_vars, min_h, max_h)
    
    probs = hist / tf.reduce_sum(hist) # divide by the number of activation samples in the current training batch

    entropy = -tf.reduce_sum(probs * tf.experimental.numpy.log2(probs+eps))
    
    return entropy, (min_h, max_h)

def calc_sparsity_regularization(inputs):
    abs_inputs = tf.abs(inputs)
    regularization_loss = tf.reduce_sum(abs_inputs)
    
    return regularization_loss

# To estimate the histogram, we first remove outliers in the
# activations if the samples are outside the range [μ−3σ; μ+3σ]
def calculate_histogram_range(variables, scale = 3):
    std = tf.math.reduce_std(variables)
    condition = tf.reduce_all(tf.equal(std, 0.0))  # Check if all elements of std are equal to 0.0
    std = tf.cond(condition, lambda: 1.0, lambda: std)
    mean = tf.math.reduce_mean(variables)
    
    return mean - std*scale, mean + std*scale

# variables (numpy.ndarray): The variables to calculate the weights for.
# min_h (float): The minimum value of the histogram bins.
# max_h (float): The maximum value of the histogram bins.
def calculate_weights(variables, min_h, max_h):
    xk = tf.reshape(variables, [-1, 1])  # Reshape variables to a column vector (65536, 1)
    t = tf.linspace(0, nbins - 1, nbins)
    w = weight(xk, t, nbins) # (65536, 256)
    deltas = tf.reduce_sum(w, axis=0)  # Compute the sum along the first axis (j)
    deltas = tf.reshape(deltas, (-1, 1)) # (256, 1)
    
    return deltas

def weight(xk, t, nbins):
    xk = tf.cast(xk, tf.float32)  # Cast xk to float32
    t = tf.cast(t, tf.float32)  # Cast t to float32
    xk = tf.abs(xk - t) # Compute the absolute difference element-wise
    mask = tf.less_equal(xk, 1)  # Create a boolean mask for elements <= 1
    w = tf.where(mask, xk, tf.zeros_like(xk))  # Use the mask to select elements or zeros
    
    return w

def calculate_histogram(variables, min_h, max_h):
    flat_vars = tf.reshape(variables, (-1,1)) # shape=(65536, 1)
    scaled_vars = (flat_vars-min_h)*(nbins - 1)/(max_h-min_h)
    calc_w = calculate_weights(scaled_vars, min_h, max_h)
    
    return calc_w

def visualize_histogram(variables):    
    min_h, max_h = calculate_histogram_range(variables)
    our_hist = calculate_histogram(variables, min_h, max_h)

    np_hist, _ = np.histogram(variables, bins=nbins, range=(min_h.numpy(), max_h.numpy())) # shape 256 * 257

    from matplotlib import pyplot as plt
    plt.plot(np_hist, "-b", label="histogram using numpy")
    plt.plot(our_hist, "-r", label="histogram using linear weight (tent function) (Jona et. al.)")
    plt.legend(loc="upper left")

    plt.title("Histogram comparison between our implementation and Numpy")

    plt.show()
    
def test_histogram():
    variables = np.random.normal(size=(10000))
    visualize_histogram(variables)

# test_histogram()
