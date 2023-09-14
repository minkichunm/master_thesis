from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model

kernel_initializer = 'he_normal'
activation = "relu"

def get_simplemodel(channels=128):
    shape=(28,28,1)
    
    inputs = Input(shape)
    layer = Flatten()(inputs)
    layer = Dense(units=channels*2, activation=activation, kernel_initializer=kernel_initializer)(layer)
    layer = Dense(units=channels, activation=activation, kernel_initializer=kernel_initializer)(layer)
    output = Dense(10, activation='linear', use_bias=True, kernel_initializer=kernel_initializer)(layer)

    model = Model(inputs, output)
    return model

# Generate a complex NN model
def get_model(chs=128):
    shape=(32, 32, 3)
    
    inputs = Input(shape)
    layer = Conv2D(chs * 2, (3, 3), padding='same', kernel_initializer=kernel_initializer)(inputs)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    
    layer = Flatten()(layer)

    output = Dense(10, activation='linear', use_bias=True, kernel_initializer=kernel_initializer)(layer)

    model = Model(inputs, output)
    return model
    
