#Readme
#pip install kaggle
#pip install -U scikit-learn
#pip install Pillow
#pip install matplotlib
#pip install pandas
#pip install pydot
#pip install plotly
#brew install graphviz

#FOR MAC: watch Readme

# STEP 0 : Import Librairies
import os
import shutil
import re
import sys
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from itertools import product
#from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from CI_ai_lib import (show_result, count_files_with_word, organize_files_by_labels,
                       create_folders_for_labels, split_files, make_predictions, plotloss,
                       get_image_dimensions, record_csv, delete_random_files,
                       get_experiment_number, generate_experiment_id, equilibrate_folders)
# import mew project
from Tools.mew.md_to_pdf import mew_md_to_pdf

#import our models
from Models.model_simple import create_model_simple  
from Models.model_medium import create_model_medium
from Models.model_complex import create_model_complex 
from Models.model_Xception_s import create_model_Xception_small



#---------------------------------------
#       META PARAMETERS
#---------------------------------------
labels              = ['dog', 'cat'] #labels that where are working on (becarefull, be sure that their is no error or if it unsupervised)
base_directory      = '.' #path of the super folder
initial_directory   = 'Dataset/train'   #Where are initially the data
experiment_number_name     = 'Experiment2/' #where do we want to create the folders containing our set for train and validation



# Définir les valeurs que vous souhaitez tester pour chaque paramètre
img_height_values   = [150]       # Exemple de différentes hauteurs d'image
img_width_values    = [150]        # Exemple de différentes largeurs d'image
batch_size_values   = [256] #[128, 64, 32, 15]    # Exemple de différentes tailles de lot
epochs_values       = [2]            #[2, 10, 20, 30]     #2,    # Exemple de différentes valeurs d'époque
learning_rate_values= [1e-4]   #[1e-4, 1e-3, 1e-2]  # Exemple de différentes taux d'apprentissage


optimizer_values = ['adam'] #['adam', 'sgd', 'rmsprop']  # Exemple de différents optimiseurs
percentage_to_delete_dog_values = [0, 25, 50, 75] #0 , 25, 50, 75      # Exemple de différentes valeurs de pourcentage à supprimer pour les chiens
percentage_to_delete_cat_values = [0, 25, 50, 75] #0, 25, 50, 75      # Exemple de différentes valeurs de pourcentage à supprimer pour les chats
splitting_values = [0.8]  # Exemple de différentes valeurs pour l'hyperparamètre "splitting"

model_values = [create_model_simple(), create_model_medium()] # , create_model_complex(), create_model_Xception_small()

# Utilisez product pour créer toutes les combinaisons possibles de paramètres
param_combinations = list(product(img_height_values, img_width_values, batch_size_values, epochs_values,
                                   learning_rate_values, optimizer_values, percentage_to_delete_dog_values,
                                   percentage_to_delete_cat_values, splitting_values, model_values))
actual_combinaison = 0

# Boucle sur toutes les combinaisons de paramètres
for params in param_combinations:
    try:
        img_height, img_width, BATCH_SIZE, EPOCHS, LEARNING_RATE, optimizer, \
        percentage_to_delete_dog, percentage_to_delete_cat, splitting, model = params


        #get the id of the experiment
        markdown_file_path = "experiment_number.md"  # Remplacez par le chemin de votre fichier Markdown
        experiment_number = get_experiment_number(markdown_file_path)
        #print("numero de l'experience: ",experiment_number)
        id_generated = generate_experiment_id()
        experiment_id = str(experiment_number)+str(id_generated)
        print("experiment id: ",experiment_id)
        final_directory     = str(experiment_number_name)+str(experiment_id) #where do we want to create the folders containing our set for train and validation
        
        #Get the number of combinaisons
        num_combinations = len(list(param_combinations)) # Comptez le nombre de combinaisons
        print(f"Nombre de combinaisons de paramètres : {num_combinations}")# Affichez le nombre de combinaisons
        print("combination:",param_combinations[actual_combinaison] )# to print all the combinaisons
        
        # Show what is which model is it
        if model == model_values[0]: 
            print("Model: Model Simple") 
            model_name = 'Experiment1_Model_Simple'
        
        if model == model_values[1]: 
            print("Model: Model Medium") 
            model_name = 'Experiment1_Model_Medium'
        """ 💡 to adjust with the models we are using    
        if model == model_values[2]: 
            print("Model: Model Complex") 
            model_name = 'Experiment1_Model_Complex'
        if model == model_values[3]: 
            print("Model: Model Complex") 
            model_name = 'Experiment1_Model_Xception_small'
        """    
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
        print("number of testing features:\t",  len(os.listdir('Dataset/Test_set')))

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
        print("number of dog images in dog folder:", len(os.listdir(os.path.join(final_directory, "dog"))))
        print("number of cat images in cat folder:", len(os.listdir(os.path.join(final_directory, "cat"))))


        # 3 equilibriaite the number of labels images
        equilibrate_folders(os.path.join(final_directory, "dog"), os.path.join(final_directory, "cat")) # Check if we have the same amount of dog images and cat images


        # 4 - Reduce the number of images in the Dataset (or not) 
        delete_random_files(folder_path = os.path.join(final_directory, "dog"), percentage_to_delete = percentage_to_delete_dog)
        delete_random_files(folder_path = os.path.join(final_directory, "cat"), percentage_to_delete = percentage_to_delete_cat)
        nbr_dog = len(os.listdir(os.path.join(final_directory, "dog")))
        nbr_cat = len(os.listdir(os.path.join(final_directory, "cat")))
        print("number of dog images in dog folder after dataset reduction:", nbr_dog)
        print("number of cat images in cat folder after dataset reduction:", nbr_cat)

        # 5 - Split the training_set and the validation_set for each label (here dog and cats)
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


        #plot_images_from_generator(train_generator, num_images=9)

        """
        plt.figure(figsize=(10, 10))
        for images, labels in train_generator.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(int(labels[i]))
                plt.axis("off")
        plt.savefig('your_image_filename.png')
        """


        #---------------------------------------
        # STEP 6 : MODEL ARCHITECTURE DESIGN
        #---------------------------------------


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
        #    callbacks = [earlystop], #,learning_rate_reduction
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
        #    workers = 4,
        #    use_multiprocessing=True,
        )

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
        report_name = "experiment_n°"+experiment_id+"_Report.md"
        report = open(os.path.join(final_directory, report_name), "w")
        report.writelines(f"## Experiment number:\t{experiment_id}\n\n") 
        report.writelines(f"Combination studied:\t*{param_combinations[actual_combinaison]}*\n\n")
        report.writelines(f"Min dimensions:\t{min_width}x{min_height}\n\n")
        report.writelines("number of dog in training set:\t{}\n\n".format(len(os.listdir(os.path.join(final_directory, "Training_set/dog")))))
        report.writelines("number of cat in training set:\t{}\n\n".format(len(os.listdir(os.path.join(final_directory, "Training_set/cat")))))
        report.writelines("number of dog in validation set:\t{}\n\n".format(len(os.listdir(os.path.join(final_directory, "Validation_set/dog")))))
        report.writelines("number of cat in validation set:\t{}\n\n".format(len(os.listdir(os.path.join(final_directory, "Validation_set/cat")))))
        report.writelines(f"number of training features:\t{len(os.listdir('Dataset/train'))}\n\n")
        report.writelines(f"number of testing features:\t{len(os.listdir('Dataset/Test_set'))}\n\n")
        report.writelines(f"Loss on validation set: {evaluation[0]}\n\n")
        report.writelines(f"Accuracy on validation set: {evaluation[1]}\n\n")
        model_plot_img = './' + os.path.join(final_directory, 'model_plot.png')
        train_val_acc_plot_img = './' + os.path.join(final_directory, 'train_val_acc_plot.png')
        train_plot_img = './' + os.path.join(final_directory, 'training_plot.png')
        # print('>>', model_plot_img, train_val_acc_plot_img, train_plot_img)
        report.writelines(f"![model_plot]({model_plot_img})\n\n")
# Issue with these last to images
        report.writelines(f"![train_val_acc_plot]({train_val_acc_plot_img})\n\n")
        report.writelines(f"![training_plot]({train_plot_img})\n\n")
        report.close()

        #Export Report as pdf
        mew_md_to_pdf(str(experiment_number_name)+experiment_id+"/experiment_n°"+experiment_id+"_Report.md", 'Tools/mew/css/style1.css', str(experiment_number_name)+experiment_id+"/experiment_n°"+experiment_id+"_Report.pdf")

        # Function to create
        #def visualize_validation_results(?):
        #def visualize_train_results(?):

        #plot training history
        show_result(history, directory = final_directory)
        plotloss(history, directory = final_directory)
        

        # Save log of the model
        print(record_csv(experiment_id = experiment_id,
            model_name = model_name,
            experiment_path = final_directory,
            splitting_values = splitting,
            save_history = history, 
            lr  = LEARNING_RATE, 
            bs  = BATCH_SIZE, 
            opt = optimizer,
            number_of_img_train = nb_train_images , 
            number_of_img_val = nb_validation_images, 
            number_of_cat = nbr_cat, 
            number_of_dog = nbr_dog))

        #save the numer of combinaison
        with open(os.path.join(final_directory, 'combinaison.md'), "w") as md_file:            
            # Écrire la première combinaison uniquement
            md_file.write(str(param_combinations[actual_combinaison]))


    except Exception as e:
        print(f"Error encountered for parameters: {params}")
        print(f"Error details: {str(e)}")
        continue  # Passe à la combinaison de paramètres suivante en cas d'erreur
    
    #---------------------------------------
    # STEP 13 : CLEAN MEMORY
    #---------------------------------------
    # Sauvegarde du modèle (optionnelle)
    shutil.rmtree(os.path.join(final_directory, "dog"))
    shutil.rmtree(os.path.join(final_directory, "cat"))
    shutil.rmtree(os.path.join(final_directory, "Training_set"))
    shutil.rmtree(os.path.join(final_directory, "Validation_set"))

    actual_combinaison = actual_combinaison + 1


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
