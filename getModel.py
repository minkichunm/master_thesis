from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Activation, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv3D, MaxPool3D, GlobalAveragePooling3D
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.applications import ResNet50, MobileNetV3Small
from tensorflow.keras.applications.densenet import DenseNet121

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

def get_3d_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = Input((width, height, depth, 1))

    layer = Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    layer = MaxPool3D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)

    layer = Conv3D(filters=64, kernel_size=3, activation="relu")(layer)
    layer = MaxPool3D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)

    layer = Conv3D(filters=128, kernel_size=3, activation="relu")(layer)
    layer = MaxPool3D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)

    layer = Conv3D(filters=256, kernel_size=3, activation="relu")(layer)
    layer = MaxPool3D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)

    layer = GlobalAveragePooling3D()(layer)
    layer = Dense(units=512, activation="relu")(layer)
    layer = Dropout(0.3)(layer)

    outputs = Dense(units=2, activation="sigmoid")(layer)

    # Define the model.
    model = Model(inputs, outputs, name="3dcnn")
    return model
    


