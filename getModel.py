from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Activation, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.applications import MobileNetV2, ResNet50, MobileNetV3Small
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.efficientnet import EfficientNetB0

kernel_initializer = 'he_normal'
activation = "relu"

def get_simplemodel(channels=128):
    shape=(28,28,1)
    
    inputs = Input(shape)
    layer = Flatten()(inputs)
    layer = Conv2D(channels * 2, (3, 3), padding='same', kernel_initializer=kernel_initializer)(inputs)
    layer = Dense(units=channels, activation=activation, kernel_initializer=kernel_initializer)(layer)
    output = Dense(10, activation='linear', use_bias=True, kernel_initializer=kernel_initializer)(layer)

    model = Model(inputs, output)
    return model
    

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
    
def get_modified_model(chs=128):
    shape = (32, 32, 3)
    
    inputs = Input(shape)
    layer = Conv2D(chs * 2, (3, 3), padding='same', kernel_initializer=kernel_initializer, strides=(2, 2))(inputs)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer, strides=(2, 2))(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer, strides=(2, 2))(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    
    layer = Flatten()(layer)

    output = Dense(10, activation='softmax', use_bias=True, kernel_initializer=kernel_initializer)(layer)

    model = Model(inputs, output)
    return model


def get_modified_cifar10_model2(chs=128):
    # Define the input layer
    shape = (32, 32, 3)
    
    inputs = Input(shape)
    
    # First set of Conv => ReLU => Conv => ReLU => MaxPooling => Dropout
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer)(inputs)
    layer = Activation(activation)(layer)
    layer = Conv2D(chs, (3, 3), kernel_initializer=kernel_initializer)(layer)
    layer = Activation(activation)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(0.25)(layer)

    # Second set of Conv => ReLU => Conv => ReLU => MaxPooling => Dropout
    layer = Conv2D(chs/2, (3, 3), padding='same', kernel_initializer=kernel_initializer)(layer)
    layer = Activation(activation)(layer)
    layer = Conv2D(chs/2, (3, 3), kernel_initializer=kernel_initializer)(layer)
    layer = Activation(activation)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(0.25)(layer)

    # Flatten => Dense => ReLU => Dropout
    layer = Flatten()(layer)
    layer = Dense(512, kernel_initializer=kernel_initializer)(layer)
    layer = Activation(activation)(layer)
    layer = Dropout(0.5)(layer)

    # Output layer with a softmax classifier
    outputs = Dense(10, activation='softmax')(layer)

    # Create the model
    model = Model(inputs, outputs)

    return model


def get_modified_cifar10_model(chs=128):
    shape = (32, 32, 3)
    inputs = Input(shape)

    # First Convolutional Layer
    layer = Conv2D(chs, (3, 3), padding='same', activation='relu')(inputs)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    # Second Convolutional Layer
    layer = Conv2D(chs, (3, 3), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    # First MaxPooling Layer
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    # Third Convolutional Layer
    layer = Conv2D(chs * 2, (3, 3), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    # Fourth Convolutional Layer
    layer = Conv2D(chs * 2, (3, 3), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    # Second MaxPooling Layer
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    # Flatten the output from the convolutional layers
    layer = Flatten()(layer)

    # First Fully Connected Layer
    layer = Dense(512, activation='relu')(layer)

    # Output Layer
    output = Dense(10, activation='softmax')(layer)

    # Create the model with the defined input and output layers
    model = Model(inputs, output)

    return model
    
def get_modified_model2(chs=128):
    shape = (32, 32, 3)
    
    # Define the input layer with the specified shape
    inputs = Input(shape)
    
    # First Convolutional Layer with Stride
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer, strides=(2, 2))(inputs)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    
    # Second Convolutional Layer with Stride
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer, strides=(2, 2))(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    
    # Third Convolutional Layer with Stride
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer, strides=(2, 2))(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    
    # Flatten the output from the convolutional layers
    layer = Flatten()(layer)

    # Dense Layer
    output = Dense(10, activation='linear', use_bias=True, kernel_initializer=kernel_initializer)(layer)

    # Create the model with the defined input and output layers
    model = Model(inputs, output)
    
    return model

def get_modified_model3(chs=128):
    shape = (32, 32, 3)
    
    # Define the input layer with the specified shape
    inputs = Input(shape)
    
    # First Convolutional Layer with MaxPooling
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer)(inputs)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    # Second Convolutional Layer with MaxPooling
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    # Third Convolutional Layer with MaxPooling
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    # Flatten the output from the convolutional layers
    layer = Flatten()(layer)

    # Dense Layer
    output = Dense(10, activation='linear', use_bias=True, kernel_initializer=kernel_initializer)(layer)

    # Create the model with the defined input and output layers
    model = Model(inputs, output)
    
    return model

def get_32model(chs=32):
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
    
def get_3model(chs=128):
    shape = (32, 32, 3)

    inputs = Input(shape)

    # First convolutional layer
    layer = Conv2D(chs * 2, (3, 3), padding='same', kernel_initializer=kernel_initializer)(inputs)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    # Flatten layer
    layer = Flatten()(layer)

    # Output layer
    output = Dense(10, activation='linear', use_bias=True, kernel_initializer=kernel_initializer)(layer)

    return Model(inputs, output)
    
    

def get_modified_cifar10_model3(chs=64):
    shape = (32, 32, 3)
    
    inputs = Input(shape)
    
    # Layer 1
    layer = Conv2D(chs, (4, 4), padding='same', activation='relu')(inputs)
    layer = BatchNormalization()(layer)
    
    # Layer 2
    layer = Conv2D(chs, (4, 4), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(0.2)(layer)
    
    # Layer 3
    layer = Conv2D(chs * 2, (4, 4), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    
    # Layer 4
    layer = Conv2D(chs * 2, (4, 4), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(0.25)(layer)
    
    # Layer 5
    layer = Conv2D(chs * 2, (4, 4), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    
    # Layer 6
    layer = Conv2D(chs * 2, (4, 4), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(0.35)(layer)
    
    layer = Flatten()(layer)
    
    # Fully Connected Layers
    layer = Dense(256, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    
    # Output Layer
    output = Dense(10, activation='softmax')(layer)

    model = Model(inputs, output)
    
    return model
    
    

def get_celeba_temp(chs=32):
    shape = (218, 178, 3)
    
    inputs = Input(shape)
    
    # Layer 1
    layer = Conv2D(chs, (3, 3), padding='same', activation='relu')(inputs)
    
    # Layer 2
    layer = Conv2D(chs * 2, (3, 3), activation='relu', padding='same')(layer)
    layer = BatchNormalization()(layer)
    
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    
    # Layer 3
    layer = Conv2D(chs * 2, (3, 3), activation='relu')(layer)
    layer = BatchNormalization()(layer)
    
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(0.5)(layer)
    
    # Layer 4
    layer = Conv2D(chs * 2, (3, 3), activation='relu', padding='same')(layer)
    layer = BatchNormalization()(layer)
    
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(0.5)(layer)
    
    # Flatten
    layer = Flatten()(layer)
    
    # Layer 5
    layer = Dense(64, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    
    # Layer 6
    layer = Dense(32, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    
    # Output Layer
    output = Dense(2, activation='softmax')(layer)

    model = Model(inputs, output)
    
    return model
    
def get_celeba_temp2(chs=64):
    shape = (218, 178, 3)
    
    inputs = Input(shape)
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer, strides=(2, 2))(inputs)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    
    layer = Conv2D(chs, (3, 3), padding='same', kernel_initializer=kernel_initializer, strides=(2, 2))(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    
    layer = Flatten()(layer)

    output = Dense(2, activation='softmax', use_bias=True, kernel_initializer=kernel_initializer)(layer)

    model = Model(inputs, output)
    
    return model
    
    
def get_modified_cifar10_model4(chs=32):
    shape = (32, 32, 3)

    inputs = Input(shape)

    # First convolutional layer
    layer = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, (3, 3), activation='relu', padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D((2, 2))(layer)

    # Second convolutional layer
    layer = Conv2D(64, (3, 3), activation='relu', padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, (3, 3), activation='relu', padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D((2, 2))(layer)

    # Third convolutional layer
    layer = Conv2D(128, (3, 3), activation='relu', padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3, 3), activation='relu', padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D((2, 2))(layer)

    # Flatten layer
    layer = Flatten()(layer)
    layer = Dropout(0.2)(layer)

    # Hidden layer
    layer = Dense(1024, activation='relu')(layer)
    layer = Dropout(0.2)(layer)

    # Output layer
    output = Dense(10, activation='softmax')(layer)

    model = Model(inputs, output)
    return model

def get_mobilenetv2(chs=256, input_shape=None, num_classes=0):
    inputs = Input(shape=input_shape)

    # Create MobileNetV2 base model
    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=input_shape,
        pooling = 'avg',
    )

    # Get the features from the base model
    layer = base_model(inputs, training = True)

    # Stack additional layers
    layer = Dense(chs, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    outputs = Dense(num_classes, activation="softmax")(layer)

    # Create the final model
    model = Model(inputs=inputs, outputs=outputs)

    return model

def get_mobilenetv3s(chs=256, input_shape=None, num_classes=0):
    inputs = Input(shape=input_shape)

    # Create mobilenetv3s base model
    base_model = MobileNetV3Small(
        weights=None,
        include_top=False,
        input_shape=input_shape,
        pooling = 'avg',
    )

    # Get the features from the base model
    layer = base_model(inputs, training = True)

    # Stack additional layers
    layer = Dense(chs, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    outputs = Dense(num_classes, activation="softmax")(layer)

    # Create the final model
    model = Model(inputs=inputs, outputs=outputs)

    return model

def get_densenet121(chs=256, input_shape=None, num_classes=0):
    inputs = Input(shape=input_shape)

    # Create DenseNet121 base model
    base_model = DenseNet121(
        weights=None,
        include_top=False,
        input_shape=input_shape,
        pooling = 'avg',
    )

    # Get the features from the base model
    layer = base_model(inputs, training = True)

    # Stack additional layers
    layer = Dense(chs, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    outputs = Dense(num_classes, activation="softmax")(layer)

    # Create the final model
    model = Model(inputs=inputs, outputs=outputs)

    return model

def get_efficientnetb0(chs=256, input_shape=None, num_classes=0):
    inputs = Input(shape=input_shape)

    # Create EfficientNetB0 base model
    base_model = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=input_shape,
        pooling = 'avg',
    )

    # Get the features from the base model
    layer = base_model(inputs, training = True)

    # Stack additional layers
    layer = Dense(chs, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    outputs = Dense(num_classes, activation="softmax")(layer)

    # Create the final model
    model = Model(inputs=inputs, outputs=outputs)

    return model
    
def get_resnet50(chs=256, input_shape=None, num_classes=0):
    inputs = Input(shape=input_shape)

    # Create Resnet50 base model
    base_model = ResNet50(
        weights=None,
        include_top=False,
        input_shape=input_shape,
        pooling = 'avg',
    )

    # Get the features from the base model
    layer = base_model(inputs, training = True)

    # Stack additional layers
    layer = Dense(chs, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    outputs = Dense(num_classes, activation="softmax")(layer)

    # Create the final model
    model = Model(inputs=inputs, outputs=outputs)

    return model


    


