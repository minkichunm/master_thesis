from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from getModel import *

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
        
        steps_per_epoch = x_train.shape[0] // batch_size
        return x_train, y_train, x_test, y_test, (x_train, y_train), steps_per_epoch

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
        df = pd.read_csv(celeba_folder+'list_attr_celeba.csv', index_col=0)
        df_partition_data = pd.read_csv(celeba_folder+'list_eval_partition.csv')
        df_partition_data.set_index('image_id', inplace=True)
        df = df_partition_data.join(df['Male'], how='inner')
        x_train, y_train = generate_df(0, 'Male', train_samples, df, celeba_folder)
        steps_per_epoch = x_train.shape[0] // batch_size
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        
        x_test, y_test = generate_df(1, 'Male', validation_samples, df, celeba_folder)
        num_class = 1
        return x_train, y_train, x_test, y_test, train_generator, steps_per_epoch
    
    else:
        raise Exception(f"Error: Invalid dataset choice '{dataset}'. Please choose a valid dataset.")    
    

def load_model_function(selected_model):
    model_functions = {
        1: get_mobilenetv3s,
        2: get_densenet121,
        3: get_resnet50,
        4: get_resnet50,
        5: get_resnet50,
        6: get_mobilenetv3s,
        7: get_modified_cifar10_model,
        8: get_modified_cifar10_model2,
        9: get_modified_cifar10_model3,
        10: get_celeba_temp,
        11: get_celeba_temp2,
        12: get_modified_model3,
        13: get_modified_cifar10_model4,
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

def representative_dataset_gen(train_images):
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    # Model has only one input so each data point has one element.
        yield [input_value]
