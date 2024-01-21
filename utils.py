from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from getModel import *
from scipy.ndimage import zoom
import nibabel as nib
from scipy import ndimage
import os
import random


def write_data_to_file(filename, data_list, step):
    with open(filename, "a") as file:
        for i in range(0, len(data_list), step):
            epoch_results = data_list[i:i + step]
            file.write(", ".join(map(str, epoch_results)))
            file.write("\n")

def write_to_file(filename, data_list):
    with open(filename, "a") as file:
        file.write(", ".join(map(str, data_list)))
        file.write("\n")
        
def load_image(file_name):
    img = load_img(file_name)
    X = img_to_array(img)
    X = X.reshape((1,) + X.shape)

    return X

def load_dataset(dataset=None, celeba_folder=None, train_samples=None, validation_samples=None, test_samples=None, batch_size=None):
    print("Loading dataset start")
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Resize images to (32, 32, 1)
        x_train_resized = zoom(x_train, (1, 1.1429, 1.1429), order=1)
        x_test_resized = zoom(x_test, (1, 1.1429, 1.1429), order=1)
        x_train_resized = np.expand_dims(x_train_resized, axis=-1)
        x_test_resized = np.expand_dims(x_test_resized, axis=-1)
        
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        train_generator = data_generator.flow(x_train_resized, y_train, batch_size)
       
        steps_per_epoch = x_train.shape[0] // batch_size
        return x_train_resized, y_train, x_test_resized, y_test, train_generator, steps_per_epoch

    elif dataset == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        train_generator = data_generator.flow(x_train, y_train, batch_size)
        steps_per_epoch = x_train.shape[0] // batch_size
        return x_train, y_train, x_test, y_test, train_generator, steps_per_epoch

    elif dataset == "celeba":
        height = 218 
        width = 178
        input_shape = (height, width, 3)
        train_samples = 4000
        validation_samples = 1000
        celeba_folder = "celeba_dataset/"
        df = pd.read_csv(celeba_folder+'list_attr_celeba.csv', index_col=0)
        df_partition_data = pd.read_csv(celeba_folder+'list_eval_partition.csv')
        df_partition_data.set_index('image_id', inplace=True)
        df = df_partition_data.join(df['Male'], how='inner')
        x_train, y_train = generate_df(0, 'Male', train_samples, df, celeba_folder)
        steps_per_epoch = x_train.shape[0] // batch_size
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        train_datagen.fit(x_train)

        train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=batch_size,
        )
        
        x_test, y_test = generate_df(1, 'Male', validation_samples, df, celeba_folder)
        
        return x_train, y_train, x_test, y_test, train_generator, steps_per_epoch
    
    elif dataset == "3d":
        # Folder "CT-0" consist of CT scans having normal lung tissue,
        # no CT-signs of viral pneumonia.
        normal_scan_paths = [
            os.path.join(os.getcwd(), "MosMedData/CT-0", x)
            for x in os.listdir("MosMedData/CT-0")
        ]
        # Folder "CT-23" consist of CT scans having several ground-glass opacifications,
        # involvement of lung parenchyma.
        abnormal_scan_paths = [
            os.path.join(os.getcwd(), "MosMedData/CT-23", x)
            for x in os.listdir("MosMedData/CT-23")
        ]

        print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
        print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths))) 
        
        # Read and process the scans.
        # Each scan is resized across height, width, and depth and rescaled.
        abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
        normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

        # For the CT scans having presence of viral pneumonia
        # assign 1, for the normal ones assign 0.
        abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
        normal_labels = np.array([0 for _ in range(len(normal_scans))])

        # Split data in the ratio 70-30 for training and validation.
        x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
        y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
        x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
        y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
        print(
            "Number of samples in train and validation are %d and %d."
            % (x_train.shape[0], x_val.shape[0])
        )        
        
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))        
        train_dataset = (
            train_loader.shuffle(len(x_train))
            .map(train_preprocessing)
            .batch(batch_size)
            .prefetch(2)
        )
        # Only rescale.
        validation_dataset = (
            validation_loader.shuffle(len(x_val))
            .map(validation_preprocessing)
            .batch(batch_size)
            .prefetch(2)
        )        
        steps_per_epoch = x_train.shape[0] // batch_size
        
        return None, None, validation_dataset, None, train_dataset, steps_per_epoch
        
        
    else:
        raise Exception(f"Error: Invalid dataset choice '{dataset}'. Please choose a valid dataset.")    
    

def load_model_function(selected_model):
    model_functions = {
        1: get_mobilenetv3s,
        2: get_densenet121,
        3: get_resnet50,
        4: get_3d_model,
    }
    try:
        if selected_model in model_functions:
            model_function = model_functions[selected_model]
        else:
            raise Exception(f"Error: Invalid model choice '{selected_model}'. Please choose a valid model.")
    except Exception as e:
        print(str(e))
        sys.exit(-1)

    return model_function
    
def generate_df(partition, attribute, nsamples, df, celeba_folder):
    images_folder = celeba_folder + 'img_align_celeba/'
    new_df = df[(df['partition'] == partition) & (df[attribute] == 1)].sample(int(nsamples/2))
    new_df = pd.concat([new_df, df[(df['partition'] == partition) & (df[attribute] == -1)].sample(int(nsamples/2))])

    # Preprocessing image and setting the target attribute in the appropriate fromat for test and validation data
    if partition!=2:
        X = np.array([load_image(images_folder + file_name) for file_name in new_df.index])
        X = X.reshape(X.shape[0], 218, 178, 3)
        y = tf.reshape((new_df[attribute].replace(-1, 0)),(-1,1))
    else:
        X = []
        y = []

        for index, target in new_df.iterrows():
            img = cv2.imread(image_folder + index)
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (width, height)).astype(np.float32) / 255.0
            img = np.expand_dims(img, axis =0)
            X.append(img)
            y.append(target[attribute])
        
    return X, y

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    
def scheduler(epoch):
    if epoch < 100:
        return 0.01
    elif epoch < 150:
        return 0.01 * 0.1
    else:
        return 0.01 * 0.1 * 0.1


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume
    
@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label
