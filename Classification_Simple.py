#Readme
#pip install kaggle
#pip install -U scikit-learn
#pip install Pillow
#pip install matplotlib
#pip install pandas
#pip install pydot
#pip install ephem
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
import matplotlib.pyplot as plt
#from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout
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
                        record_csv, delete_random_files, \
                        get_experiment_number, generate_experiment_id

#---------------------------------------
#		META PARAMETERS
#---------------------------------------
img_height      = 150
img_width       = 150
channel         = 3
BATCH_SIZE 		= 15 #32
EPOCHS 			= 2 #30 #10
LEARNING_RATE 	= 1e-4
splitting 		= 0.8 # How do we want to split our training and validation set
#label_size	#How much of the dataset we want to keep
#opt #optimizer # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/legacy/Adam
# ADAM, SGD, RMSprop, Adagrad, Adadelta, Nadam, FTRL, LBFGS, Rprop, SGD with Momentum
labels              = ['dog', 'cat'] #labels that where are working on (becarefull, be sure that their is no error or if it unsupervised)
base_directory      = '.' #path of the super folder
initial_directory   = 'Dataset/train'   #Where are initially the data
final_directory     = 'Experiment1' #where do we want to create the folders containing our set for train and validation
percentage_to_delete_dog = percentage_to_delete_cat = 0 #50 # Number of cat and dog wa want to delete

markdown_file_path = "experiment_number.md"  # Remplacez par le chemin de votre fichier Markdown
experiment_number = get_experiment_number(markdown_file_path)
#print("numero de l'experience: ",experiment_number)
id_generated = generate_experiment_id()
experiment_id = str(experiment_number)+str(id_generated)
print("experiment id: ",experiment_id)
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
print("number of testing features:\t", 	len(os.listdir('Dataset/Test_set')))

# Verify the number of dog and cat image
#---- 1. It could happen that whe have wrong label images 
#---- 2. Do we have same number of images for dog and cats?
#---- 3. How changing the number of dog images will affect the accuracy? 
nbr_dog = count_files_with_word(DATASET_DIR, "Dog")
nbr_cat = count_files_with_word(DATASET_DIR, "Cat")
print(f"Number of images labeld as 'Dog' in dataset : {nbr_dog}")
print(f"Number of images labeld as 'Cat' in dataset : {nbr_cat}")


#---------------------------------------
# STEP 2: KNOW MORE ABOUT OUR DATA
#---------------------------------------
min_width, min_height = get_image_dimensions(directory = "Dataset/Test_set")
print(f"Dimension minimale en largeur (width) : {min_width}")
print(f"Dimension minimale en hauteur (height) : {min_height}")


#---------------------------------------
# STEP 3 : SPLIT TRAINING SET AND VALIDATION SET
#---------------------------------------

# 1 - Create folders for Training_set and validation_set and all folders intricated by the labels for each feature 
create_folders_for_labels(labels, final_directory)


# 2 - Create a folders for each feature containing all images labeled as it (here dog and cat folders)
organize_files_by_labels(labels, initial_directory, final_directory)
print("number of dog images in dog folder:", len(os.listdir(os.path.join(final_directory, "Dog"))))
print("number of cat images in cat folder:", len(os.listdir(os.path.join(final_directory, "Cat"))))

# 3 - Reduce the number of images in the Dataset (or not) 
delete_random_files(folder_path = os.path.join(final_directory, "Dog"), percentage_to_delete = percentage_to_delete_dog)
delete_random_files(folder_path = os.path.join(final_directory, "Cat"), percentage_to_delete = percentage_to_delete_cat)
nbr_dog = len(os.listdir(os.path.join(final_directory, "Dog")))
nbr_cat = len(os.listdir(os.path.join(final_directory, "Cat")))
print("number of dog images in dog folder after dataset reduction:", nbr_dog)
print("number of cat images in cat folder after dataset reduction:", nbr_cat)

# 4 - Split the training_set and the validation_set for each label (here dog and cats)
split_files(source_directory = os.path.join(final_directory, "Dog"), destination_directory_1 = os.path.join(final_directory, "Training_set/Dog"), destination_directory_2 = os.path.join(final_directory, "Validation_set/Dog"), pourcentage = splitting)
split_files(source_directory = os.path.join(final_directory, "Cat"), destination_directory_1 = os.path.join(final_directory, "Training_set/Cat"), destination_directory_2 = os.path.join(final_directory, "Validation_set/Cat"), pourcentage = splitting)

#-- To verify : 
print("number of dog in training set:",     len(os.listdir(os.path.join(final_directory, "Training_set/dog"))))
print("number of cat in training set:",     len(os.listdir(os.path.join(final_directory, "Training_set/cat"))))
print("number of dog in validation set:",   len(os.listdir(os.path.join(final_directory, "Validation_set/dog"))))
print("number of cat in validation set:",   len(os.listdir(os.path.join(final_directory, "Validation_set/cat"))))


#---------------------------------------
# STEP 4 : DATA AUGMENTATION
#---------------------------------------

# Préparation des données sans augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255.) # Les valeurs des canaux RVB sont dans la plage [0, 255] . Ce n'est pas idéal pour un réseau neuronal ; en général, vous devriez chercher à rendre vos valeurs d'entrée petites. Ici, vous allez normaliser les valeurs pour qu'elles soient dans la plage [0, 1]
validation_datagen = ImageDataGenerator(rescale=1.0/255.) # We should even add a normalization layer actually in Keras. normalization_layer = tf.keras.layers.Rescaling(1./255)



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
# STEP 6 : MODEL ARCHITECTURE DESIGN
#---------------------------------------

# Création du modèle


#---DUMB-----------

"""
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
"""

#---COMPLEX-----------

"""
model = Sequential()

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
"""


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


#---------------------------------------
# STEP 7 : MODEL VIZUALISATION 
#---------------------------------------
model.summary() #plot a summary of the model


plot_model(model, to_file = os.path.join(final_directory, 'model_plot.png'), show_shapes=True, show_layer_names=True) #plot the model

#---------------------------------------
# STEP 8 : MODEL COMPILING
#---------------------------------------
model.compile(
    optimizer = tf.keras.optimizers.legacy.RMSprop(
        learning_rate = LEARNING_RATE, 
    	#rho=0.9,
        #momentum=0.0,
        #epsilon=1e-07,
        #centered=False,
        #name='RMSprop',
        #**kwargs
    ),
	loss='binary_crossentropy', 
	metrics=['acc'], #tf.keras.metrics.BinaryAccuracy()
    #loss_weights=None,
    #weighted_metrics=None,
    #run_eagerly=None,
    #steps_per_execution=None,
    #jit_compile=None,
    #pss_evaluation_shards=0,
    #**kwargs
    )


#---------------------------------------
# STEP 9 : MODEL TRAINING
#---------------------------------------

# Callback initialization
#This callback will adjust the learning rate  when there is no improvement in the loss for two consecutive epochs. No need for GRID or NEAT search 
#earlystop = EarlyStopping(patience = 5)
#learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 4 ,verbose = 1, factor = 0.5, min_lr = 0.00001) 
#tf.keras.callbacks.CSVLogger('train_log.csv', separator=",", append=False)

history = model.fit(
    train_generator, #x, Y
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
#    verbose="auto",
#    callbacks = [earlystop,learning_rate_reduction],
#    validation_split=0.0,
    validation_data=validation_generator,
#    shuffle=True,
#    class_weight=None,
#    sample_weight=None,
#    initial_epoch=0,
    steps_per_epoch=nb_train_images//BATCH_SIZE,
    validation_steps=nb_validation_images//BATCH_SIZE,
#    validation_batch_size=None,
#    validation_freq=1,
#    max_queue_size=10,
#    workers=1,
#    use_multiprocessing=False,


                    )

show_result(history, directory = final_directory)
plotloss(history, directory = final_directory)

#---------------------------------------
# STEP 10 : MODEL EVALUATION
#---------------------------------------
print("\nMODEL EVALUATION ------------")

evaluation = model.evaluate(validation_generator
#    x=None,
#    y=None,
#    batch_size=None,
#    verbose="auto",
#    sample_weight=None,
#    steps=None,
#    callbacks=None,
#    max_queue_size=10,
#    workers=1,
#    use_multiprocessing=False,
#    return_dict=False,
#    **kwargs
    )



print("Loss on validation set:", evaluation[0])
print("Accuracy on validation set:", evaluation[1])

# Save log of the model
print(record_csv(experiment_id = experiment_id,
    experiment_path = final_directory,
    save_history = history, 
    lr = LEARNING_RATE, 
    bs = BATCH_SIZE, 
    opt='ADAM',
    number_of_img_train = nb_train_images , 
    number_of_img_val = nb_validation_images, 
    number_of_cat = nbr_cat, 
    number_of_dog = nbr_dog))




#---------------------------------------
# STEP 11 : MODEL PREDICTION
#---------------------------------------

make_predictions(model, test_directory = 'Dataset/Test_set', output_csv = os.path.join(final_directory, 'predictions.csv'),img_height = img_height, img_width = img_width) 


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
# Show correlation graph between all the parameters and the accuracy
# plot images where the code gets wrong (show_false_prediction)
# add remarks

# Function to create
#def visualize_validation_results(?):
#def visualize_train_results(?):


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
