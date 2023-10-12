import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def create_model_complex(img_height = 150, img_width = 150, channel = 3):


    tf.random.set_seed(42)
    model = Sequential()
    
    #model.add(Input(input_shape = (img_height, img_width, channel))),
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, channel)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (3,3), strides =(2,2))),
    model.add(Conv2D(256, kernel_size=(5,5), padding='same', activation = 'relu')),
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (3,3), strides=(2,2))),
    
    model.add(Conv2D(384, kernel_size=(3,3),padding='same', activation = 'relu')),
    model.add(Conv2D(384, kernel_size=(3,3), padding='same', activation = 'relu')),
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation = 'relu')),
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (3,3), strides=(2,2))),
    
    
    model.add(Flatten()),
    model.add(Dense(4096, activation = 'relu')),
    model.add(Dropout(rate = 0.5)),
    model.add(Dense(4096, activation = 'relu')),
    model.add(Dropout(rate = 0.5)),
    model.add(Dense(1, activation = 'sigmoid'))

    return model
