import os
import shutil
import re
import random
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
#%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt


def count_files_with_word(directory, word):
	"""
		Count number of files containing a certain word in a folder, for exemple how many images are lebelised as"dog".
	"""
	count = 0
	for filename in os.listdir(directory):
		if word in filename:
			count += 1
	return count



def move_files_with_word(source_directory, target_directory, word):
	"""
		Create a folder containing only the files corresponding the label targeted
	"""
	# Création du répertoire cible avec la première lettre en majuscule
	target_subdirectory = os.path.join(target_directory, word.capitalize())
	os.makedirs(target_subdirectory, exist_ok=True)

	# Expression régulière pour rechercher le mot dans le nom des fichiers
	word_pattern = re.compile(rf'.*{word}.*', re.IGNORECASE)

	# Parcours des fichiers dans le répertoire source
	for filename in os.listdir(source_directory):
	    source_file_path = os.path.join(source_directory, filename)

	    # Vérification si le nom du fichier contient le mot spécifique
	    if re.match(word_pattern, filename):
	        # Déplacement du fichier vers le répertoire cible
	        target_file_path = os.path.join(target_subdirectory, filename)
	        shutil.move(source_file_path, target_file_path)
	        print(f"Fichier déplacé : {filename} vers {target_subdirectory}")


import os
import random
import shutil
#crée un fichier par label
def organize_files_by_labels(labels, initial_directory, final_directory):
    # Crée le répertoire final s'il n'existe pas
    os.makedirs(final_directory, exist_ok=True)

    # Pour chaque label, crée un sous-répertoire portant le nom du label dans le répertoire final
    for label in labels:
        label_dir = os.path.join(final_directory, label)
        os.makedirs(label_dir, exist_ok=True)

        # Liste des fichiers dans le répertoire initial
        files = os.listdir(initial_directory)

        # Parcourez les fichiers et déplacez-les dans le sous-répertoire correspondant
        for file in files:
            if label in file:
                src = os.path.join(initial_directory, file)
                dst = os.path.join(label_dir, file)
                shutil.move(src, dst)


def create_folders_for_labels(labels, base_directory):
    # Crée les dossiers Training_set et Validation_set s'ils n'existent pas
    training_set_dir = os.path.join(base_directory, 'Training_set')
    validation_set_dir = os.path.join(base_directory, 'Validation_set')
    os.makedirs(training_set_dir, exist_ok=True)
    os.makedirs(validation_set_dir, exist_ok=True)

    # Crée un sous-répertoire pour chaque label dans Training_set et Validation_set
    for label in labels:
        training_label_dir = os.path.join(training_set_dir, label)
        validation_label_dir = os.path.join(validation_set_dir, label)
        os.makedirs(training_label_dir, exist_ok=True)
        os.makedirs(validation_label_dir, exist_ok=True)



def split_files(source_directory, destination_directory_1, destination_directory_2, pourcentage):
    # Vérifiez si les répertoires de destination existent, sinon, créez-les
    os.makedirs(destination_directory_1, exist_ok=True)
    os.makedirs(destination_directory_2, exist_ok=True)

    # Liste des fichiers dans le répertoire source
    files = os.listdir(source_directory)

    # Mélangez la liste des fichiers
    random.shuffle(files)

    # Calculez le nombre de fichiers à déplacer dans le premier sous-répertoire
    num_files_1 = int(len(files) * pourcentage)

    # Déplacez les fichiers dans le premier sous-répertoire
    for file_name in files[:num_files_1]:
        src = os.path.join(source_directory, file_name)
        dst = os.path.join(destination_directory_1, file_name)
        shutil.move(src, dst)

    # Déplacez les fichiers restants dans le deuxième sous-répertoire
    for file_name in files[num_files_1:]:
        src = os.path.join(source_directory, file_name)
        dst = os.path.join(destination_directory_2, file_name)
        shutil.move(src, dst)

def make_predictions(model, test_directory, output_csv,img_height, img_width):
    # Créez une liste pour stocker les résultats
    results = []

    # Parcourez les fichiers dans le répertoire de test
    for filename in os.listdir(test_directory):
        # Chargez l'image et prétraitez-la de la même manière que pour l'entraînement
        img = load_img(os.path.join(test_directory, filename), target_size=(img_height, img_width))
        img = img_to_array(img)
        img = img / 255.0  # Normalisez l'image comme lors de l'entraînement
        img = np.expand_dims(img, axis=0)  # Ajoutez une dimension pour le lot (batch)

        # Faites la prédiction
        prediction = model.predict(img)[0][0]  # Obtenez la valeur de prédiction (0 ou 1)

        # Obtenez l'ID à partir du nom du fichier en supprimant l'extension
        image_id = os.path.splitext(filename)[0]

        # Ajoutez les résultats à la liste
        results.append({'id': image_id, 'labels': int(round(prediction))})

    # Créez un DataFrame Pandas à partir des résultats
    results_df = pd.DataFrame(results)
    results_df['id'] = results_df['id'].astype('int')
    results_df = results_df.sort_values(by=['id'])

    # Enregistrez le DataFrame dans un fichier CSV
    results_df.to_csv(output_csv, index=False)

def show_result(history):

    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    print('Last train accuracy: %s'%history.history['acc'][-1])
    print('Last validation accuracy: %s'%history.history['val_acc'][-1])

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    epochs = range(1, len(loss) + 1)

    # Define a subplot
    fig, axs = plt.subplots(1,2,figsize=(15,4))

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    loss_plot = axs[0]

    loss_plot.plot(epochs, loss, 'c--', label='Training loss')
    loss_plot.plot(epochs, val_loss, 'b', label='Validation loss')
    loss_plot.set_title('Training and validation loss')
    loss_plot.set_xlabel('Epochs')
    loss_plot.set_ylabel('Loss')
    loss_plot.legend()

    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    acc_plot = axs[1]

    acc_plot.plot(epochs, acc, 'c--', label='Training acc')
    acc_plot.plot(epochs, val_acc, 'b', label='Validation acc')
    acc_plot.set_title('Training and validation accuracy')
    acc_plot.set_xlabel('Epochs')
    acc_plot.set_ylabel('Accuracy')
    acc_plot.legend()

    plt.savefig('training_plot.png')
    plt.show()

def plotloss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation', 'Accuracy'])
    plt.show()
    #plotloss(history)
