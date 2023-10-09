#Readme
#pip install kaggle
#pip install -U scikit-learn
#pip install Pillow
#pip install matplotlib
#pip install pandas
#pip install pydot
#brew install graphviz

#FOR MAC: watch Readme

# STEP 0 : Import Librairies
import os
import shutil
import re
import random
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from CI_ai_lib import show_result, \
						count_files_with_word, \
						move_files_with_word , \
						organize_files_by_labels,\
						create_folders_for_labels,\
						split_files, \
						make_predictions, \
						plotloss, \
						get_image_dimensions, \
                        record_csv

#---------------------------------------
#		META PARAMETERS
#---------------------------------------
img_height      = 150
img_width       = 150
channel         = 3

BATCH_SIZE 		= 32
EPOCHS 			= 2
LEARNING_RATE 	= 0.001
splitting 		= 0.7 # How do we want to split our training and validation set
#label_size	#How much of the dataset we want to keep
#opt #optimizer # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/legacy/Adam
# ADAM, SGD, RMSprop, Adagrad, Adadelta, Nadam, FTRL, LBFGS, Rprop, SGD with Momentum
labels              = ['dog', 'cat'] #labels that where are working on (becarefull, be sure that their is no error or if it unsupervised)
base_directory      = '.' #path of the super folder
initial_directory   = 'Dataset/train'   #Where are initially the data
final_directory     = 'Experiment1' #where do we want to create the folders containing our set for train and validation


# ------------------------------- FUNCTIONS -----------------------------


# ------------------------------- CODE -----------------------------

#---------------------------------------
# STEP 1: Data Loading 
#---------------------------------------
# download data
#kaggle competitions download -c dogs-vs-cats

# get info about our data 
dataset_dir = 'Dataset' #chemin_vers_votre_repertoire
DATASET_DIR = os.path.join(dataset_dir, 'train')

# Check if we find our data
print("number of Total images:\t", len(os.listdir('Dataset/train')))
print("number of testing features:\t", 	len(os.listdir('Dataset/test1')))

# Verify the number of dog and cat image
#---- 1. It could happen that whe have wrong label images 
#---- 2. Do we have same number of images for dog and cats?
#---- 3. How changing the number of dog images will affect the accuracy? 
nbr_dog = count_files_with_word(DATASET_DIR, "dog")
nbr_cat = count_files_with_word(DATASET_DIR, "cat")
print(f"Number of images labeld as 'dog' in dataset : {nbr_dog}")
print(f"Number of images labeld as 'cat' in dataset : {nbr_cat}")


#---------------------------------------
# STEP 2: KNOW MORE ABOUT OUR DATA
#---------------------------------------
min_width, min_height = get_image_dimensions(directory = "Dataset/test1")
print(f"Dimension minimale en largeur (width) : {min_width}")
print(f"Dimension minimale en hauteur (height) : {min_height}")

#---------------------------------------
# STEP 3 : SPLIT TRAINING SET AND VALIDATION SET
#---------------------------------------

# 1 - Create folders for Training_set and validation_set and all folders intricated by the labels for each feature 
create_folders_for_labels(labels, final_directory)


# 2 - Create a folders for each feature containing all images labeled as it (here dog and cat folders)
organize_files_by_labels(labels, initial_directory, final_directory)
print("number of dog images in dog folder:", len(os.listdir(os.path.join(final_directory, "dog"))))
print("number of cat images in cat folder:", len(os.listdir(os.path.join(final_directory, "cat"))))


# 3 - Split the training_set and the validation_set for each label (here dog and cats)
split_files(source_directory = os.path.join(final_directory, "dog"), destination_directory_1 = os.path.join(final_directory, "Training_set/dog"), destination_directory_2 = os.path.join(final_directory, "Validation_set/dog"), pourcentage = splitting)
split_files(source_directory = os.path.join(final_directory, "cat"), destination_directory_1 = os.path.join(final_directory, "Training_set/cat"), destination_directory_2 = os.path.join(final_directory, "Validation_set/cat"), pourcentage = splitting)

#-- To verify : 
print("number of dog in training set:",     len(os.listdir(os.path.join(final_directory, "Training_set/dog"))))
print("number of cat in training set:",     len(os.listdir(os.path.join(final_directory, "Training_set/cat"))))
print("number of dog in validation set:",   len(os.listdir(os.path.join(final_directory, "Validation_set/dog"))))
print("number of cat in validation set:",   len(os.listdir(os.path.join(final_directory, "Validation_set/cat"))))


#---------------------------------------
# STEP 4 : DATA AUGMENTATION
#---------------------------------------

# Préparation des données sans augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255.) #Because here rescale = 1.0, it means that we are not doing data augmentation for the moment
validation_datagen = ImageDataGenerator(rescale=1.0/255.)



#---------------------------------------
# STEP 5 : ASSIGN TRAIN_SET AND VALIDATION_SET
#---------------------------------------
TRAINING_DIR = os.path.join(final_directory, "Training_set/")
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,#X_train,y_train,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary',
                                                    target_size=(img_height, img_width),
                                                    seed=0,
                                                    shuffle=False)


VALIDATION_DIR = os.path.join(final_directory, "Validation_set/")
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, #X_val,y_val,
                                                              batch_size=BATCH_SIZE,
                                                              class_mode='binary',
                                                              target_size=(img_height, img_width),
                                                              shuffle=False)

nb_train_images     = len(os.listdir(os.path.join(final_directory, "Training_set/dog/")))   + len(os.listdir(os.path.join(final_directory, "Training_set/cat/")))
nb_validation_images= len(os.listdir(os.path.join(final_directory, "Validation_set/dog/"))) + len(os.listdir(os.path.join(final_directory, "Validation_set/dog/")))
print("\nnumber of images in the Training_set:", nb_train_images)
print("number of images in the Validation_set:", nb_validation_images)


#---------------------------------------
# STEP 6 : MODEL DESIGN
#---------------------------------------

# Création du modèle


#---DUMB-----------

"""
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, channel)))
model.add(MaxPooling2D(2, 2))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(2, 2))
#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D(2, 2))
model.add(Flatten())
#model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Dog or not a dog
"""

#---COMPLEX-----------

model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(img_height, img_width, channel)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1,activation='softmax'))


#---------------------------------------
# STEP 7 : MODEL VIZUALISATION 
#---------------------------------------
model.summary() #plot a summary of the model


plot_model(model, to_file = os.path.join(final_directory, 'model_plot.png'), show_shapes=True, show_layer_names=True) #plot the model

#---------------------------------------
# STEP 8 : MODEL COMPILING
#---------------------------------------
model.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE, 
	#rho=0.9,
    #momentum=0.0,
    #epsilon=1e-07,
    #centered=False,
    #name='RMSprop',
    #**kwargs
    ),
	loss='binary_crossentropy', 
	metrics=['acc'])



#---------------------------------------
# STEP 9 : MODEL TRAINING
#---------------------------------------
# Entraînement du modèle
history = model.fit(train_generator,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=nb_train_images//BATCH_SIZE, # 2000 images = batch_size * steps nb_images_train/batch_size
                    validation_data=validation_generator,
                    validation_steps=nb_validation_images//BATCH_SIZE, # batch Il faudra remplacer par
                    )

show_result(history, directory = final_directory)
plotloss(history, directory = final_directory)

#---------------------------------------
# STEP 10 : MODEL EVALUATION
#---------------------------------------
print("\nMODEL EVALUATION ------------")

evaluation = model.evaluate(validation_generator)
print("Loss on validation set:", evaluation[0])
print("Accuracy on validation set:", evaluation[1])

print(record_csv(history, LEARNING_RATE, BATCH_SIZE, 'ADAM',number_of_img_train = nb_train_images , number_of_img_val = nb_validation_images))

#---------------------------------------
# STEP 11 : MODEL PREDICTION
#---------------------------------------

make_predictions(model, test_directory = 'Dataset/test1', output_csv = os.path.join(final_directory, 'predictions.csv'),img_height = img_height, img_width = img_width) 


#---------------------------------------
# STEP 12 : MODEL SAVING
#---------------------------------------
# Sauvegarde du modèle (optionnelle)
model.save(os.path.join(final_directory, 'model_dogs_vs_cats_no_augmentation.h5'))

#---------------------------------------
#       GENERATE REPORT
#---------------------------------------
# call mew etc
# show the number of the experiment 
# show hyper parameters and accuracy and image size
# show time taken
# Show graph Architecture
# show number of features
# plot how many dogs and how many cats
# plot how many features in the training set and how many in the validation set
# show how many data augmentation and samples
# plot graph of plotloss
# plot result of record_csv
# Show correlation graoh between all the parameters and the accuracy
# plot images where the code gets wrong (show_false_prediction)
# add remarks


#---------------------------------------
# STEP 13 : CLEAN MEMORY
#---------------------------------------
# Sauvegarde du modèle (optionnelle)
shutil.rmtree(os.path.join(final_directory, "dog"))
shutil.rmtree(os.path.join(final_directory, "cat"))
shutil.rmtree(os.path.join(final_directory, "Training_set"))
shutil.rmtree(os.path.join(final_directory, "Validation_set"))


"""
# ========== DATA AUGMENTATION
# Préparation des données avec augmentation des images (data augmentation)
train_data_generator = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Mise à l'échelle des pixels entre 0 et 1
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
"""
