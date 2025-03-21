import os
import shutil
import re
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import markdown_it
import ephem
import datetime
import plotly.graph_objects as go
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from plotly.subplots import make_subplots
#%matplotlib inline




def get_image_dimensions(directory):
    # Initialisez les dimensions minimales avec des valeurs élevées
    min_width = float('inf')
    min_height = float('inf')

    # Parcourez les fichiers dans le répertoire
    for filename in os.listdir(directory):
        # Vérifiez si le fichier est une image (vous pouvez ajouter plus d'extensions si nécessaire)
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # Chemin complet du fichier image
            image_path = os.path.join(directory, filename)

            # Ouvrez l'image avec PIL
            img = Image.open(image_path)

            # Obtenez les dimensions de l'image
            width, height = img.size

            # Mettez à jour les dimensions minimales si nécessaire
            if width < min_width:
                min_width = width
            if height < min_height:
                min_height = height

    return min_width, min_height


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
	        shutil.copy(source_file_path, target_file_path)
	        print(f"Fichier déplacé : {filename} vers {target_subdirectory}")



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
                shutil.copy(src, dst)


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
        shutil.copy(src, dst)

    # Déplacez les fichiers restants dans le deuxième sous-répertoire
    for file_name in files[num_files_1:]:
        src = os.path.join(source_directory, file_name)
        dst = os.path.join(destination_directory_2, file_name)
        shutil.copy(src, dst)





def delete_random_files(folder_path, percentage_to_delete):
    if not os.path.exists(folder_path):
        print(f"Le dossier '{folder_path}' n'existe pas.")
        return

    if percentage_to_delete < 0 or percentage_to_delete > 100:
        print("Le pourcentage doit être compris entre 0 et 100.")
        return

    files_to_delete = []

    # Liste de tous les fichiers dans le dossier 'chat'

    if os.path.exists(folder_path):
        cat_files = os.listdir(folder_path)

        # Calculer le nombre de fichiers à supprimer en fonction du pourcentage
        num_files_to_delete = int(len(cat_files) * (percentage_to_delete / 100))

        # Sélectionner au hasard les fichiers à supprimer
        files_to_delete = random.sample(cat_files, num_files_to_delete)

    # Supprimer les fichiers sélectionnés
    for file_to_delete in files_to_delete:
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)
        print(f"Fichier supprimé : {file_path}")




# know the number of the experiment
def get_experiment_number(markdown_file_path):
    # Fonction pour extraire le numéro d'expérience de la première ligne du fichier Markdown
    def extract_experiment_number(markdown_content):
        lines = markdown_content.split("\n")
        if lines:
            first_line = lines[0]
            match = re.search(r"\d+", first_line)
            if match:
                return int(match.group())
        return None

    # Fonction pour incrémenter le numéro d'expérience dans le fichier Markdown
    def increment_experiment_number(markdown_content):
        experiment_number = extract_experiment_number(markdown_content)
        if experiment_number is not None:
            new_experiment_number = experiment_number + 1
            updated_content = re.sub(r"\d+", str(new_experiment_number), markdown_content, count=1)
            return updated_content
        return markdown_content

    # Lire le contenu du fichier Markdown
    with open(markdown_file_path, "r") as file:
        markdown_content = file.read()

    # Incrémenter le numéro d'expérience et obtenir le contenu mis à jour
    updated_content = increment_experiment_number(markdown_content)

    # Enregistrer le contenu mis à jour dans le fichier Markdown
    with open(markdown_file_path, "w") as file:
        file.write(updated_content)

    # Afficher le numéro d'expérience mis à jour
    updated_experiment_number = extract_experiment_number(updated_content)
    #print("Numéro d'expérience :", updated_experiment_number - 1)
    return updated_experiment_number - 1



def generate_experiment_id():
    # Obtenez la première lettre du jour actuel (par exemple, "L" pour "Lundi")
    current_day = datetime.datetime.now().strftime("%A")[0]
    
    # Obtenez l'heure actuelle au format HH (heures) et MM (minutes)
    current_time = datetime.datetime.now().strftime("%H%M")
    
    # Obtenez la position actuelle de la lune
    moon_position = ephem.Moon()
    moon_position.compute()
    moon_phase = int(moon_position.phase / 100)  # Obtenez la phase de la lune sous forme de nombre entier

    # Créez la chaîne au format "-DHHccMM"
    result_string = f"-{current_day}{current_time}{moon_phase:02d}"
    
    return result_string

def equilibrate_folders(folder1,folder2):
    dog_folder = folder1
    cat_folder = folder2

    # Compter le nombre de fichiers dans chaque dossier
    num_dog_files = len(os.listdir(dog_folder))
    num_cat_files = len(os.listdir(cat_folder))

    while num_dog_files != num_cat_files:
        
        if num_dog_files > num_cat_files:
            # Supprimer un fichier du dossier 'Dog' si nécessaire
            dog_files = os.listdir(dog_folder)
            os.remove(os.path.join(dog_folder, dog_files[0]))
        else:
            # Supprimer un fichier du dossier 'Cat' si nécessaire
            cat_files = os.listdir(cat_folder)
            os.remove(os.path.join(cat_folder, cat_files[0]))

        # Mettre à jour le nombre de fichiers dans chaque dossier
        num_dog_files = len(os.listdir(dog_folder))
        num_cat_files = len(os.listdir(cat_folder))

    print("Folders 'Dog' and 'Cat' have now the same amount of files.")




#----------------------------------------------------------------------- MAKE PREDECTION
def make_predictions(model, test_directory, output_csv, img_height, img_width):
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
        prediction = model.predict(img)[0][0]  # Obtenez la valeur de prédiction (entre 0 et 1)

        # Obtenez l'ID à partir du nom du fichier en supprimant l'extension
        image_id = os.path.splitext(filename)[0]

        # Mappez la prédiction à 1 (dog) ou 0 (cat)
        if prediction >= 0.5:
            label = 1  # (dog)
        else:
            label = 0  # (cat)

        # Ajoutez les résultats à la liste
        results.append({'id': image_id, 'labels': label})

    # Créez un DataFrame Pandas à partir des résultats
    results_df = pd.DataFrame(results)
    results_df['id'] = results_df['id'].astype('int')
    results_df = results_df.sort_values(by=['id'])

    # Enregistrez le DataFrame dans un fichier CSV
    results_df.to_csv(output_csv, index=False)



#----------------------------------------------------------------------- EVALUATE
def record_csv(experiment_id,model_name,experiment_path,splitting_values,save_history,lr,bs,opt,number_of_img_train , number_of_img_val, number_of_cat, number_of_dog ):
  acc = save_history.history['acc']
  val_acc = save_history.history['val_acc']
  loss  = save_history.history['loss']
  val_loss  = save_history.history['val_loss']

  df_acc   = pd.DataFrame(acc)
  df_val_acc = pd.DataFrame(val_acc)
  df_loss = pd.DataFrame(loss)
  df_val_loss = pd.DataFrame(val_loss)
  df_fitting_rate = df_loss/df_val_loss
  #df_fitting_state = df_loss/df_val_loss

  result = pd.concat([df_acc, df_val_acc, df_loss,df_val_loss, df_fitting_rate], axis=1, join="inner")
  result["df_fitting_state"] = ""

  result.columns = ['acc','val_acc','loss','val_loss','fitting_rate','fitting_state']

  result.loc[result.fitting_rate >= float(0.8),'fitting_state']= 'Underfitting'
  result.loc[result.fitting_rate <= float(0.4),'fitting_state']= '0verfitting'
  result.loc[(result.fitting_rate >= float(0.4)) & (result.fitting_rate <= float(0.8)),'fitting_state']='Finefitting'

  result["epochs"] = result.index
  result["epochs"] = result["epochs"]+1

  result["optimizer"] = opt
  result["learning_rate"] = lr
  result["batchsize"] = bs
  result["number_of_img_train"] = number_of_img_train
  result["number_of_img_val"] = number_of_img_val
  result["nbr_cat"] = number_of_cat
  result["nbr_dog"] = number_of_dog
  result["experiment_id"] = experiment_id
  result["model_name"] = model_name
  result["splitting_values"] = splitting_values

  experiment_id = str(experiment_id)
  filename = "experiment_n°"+experiment_id+"_records.csv"
  # saving as a CSV file
  df = pd.DataFrame(result)
  df.to_csv(os.path.join(experiment_path, filename), sep =';')


  return result



#----------------------------------------------------------------------- VIZUALIZE RESULTS
def show_result(history, directory):

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

    plt.savefig(os.path.join(directory,'training_plot.png'))
    #plt.show()

def plotloss(history, directory):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation', 'Accuracy'])
    plt.savefig(os.path.join(directory, 'train_val_acc_plot.png'))
    #plt.show()


#----------------------------------------------------------------------- TO IMPLEMENT

"""
def count_files_with_word(folder_path, word_to_count):
    count = 0

    # Walk through the directory tree
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file name contains the word_to_count
            if word_to_count in file:
                count += 1

    return count

# Example usage:
folder_name = "Keras_Dataset"
word_to_count = "cat"

count = count_files_with_word(folder_name, word_to_count)
print(f"Number of files containing '{word_to_count}': {count}")
"""




