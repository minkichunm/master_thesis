from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

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
