# STEP 0: Import Libraries
import os
import shutil
import re
import tensorflow as tf
import pathlib
#from tensorflow.keras import layers, models


# Dataset URL
dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

# Get the current directory
current_directory = pathlib.Path.cwd()

# Use a relative path to save the file in the current directory
archive = tf.keras.utils.get_file(fname="./kagglecatsanddogs_5340.zip", origin=dataset_url, extract=True, cache_dir=current_directory)

# Get the directory path by removing the file extension
data_dir = pathlib.Path(archive).with_suffix('')
print("The downloaded folder is located at:", data_dir)

#-------------------- DATA FOLDERS ARCHITECTURE

# Rename the "datasets" folder to "initial_dataset"
if os.path.exists("datasets"):
    os.rename("datasets", "initial_dataset")

# Create the "Dataset/train" folder if it doesn't exist
train_dir = "Dataset/train"
os.makedirs(train_dir, exist_ok=True)

# Path to the "initial_dataset/PetImages/Cat" and "initial_dataset/PetImages/Dog" folders
cat_source_dir = "initial_dataset/PetImages/Cat"
dog_source_dir = "initial_dataset/PetImages/Dog"

# Copy files from "initial_dataset/PetImages/Cat" to "Dataset/train"
for filename in os.listdir(cat_source_dir):
    if filename.endswith(".jpg"):
        shutil.copy(os.path.join(cat_source_dir, filename), os.path.join(train_dir, filename))

# Copy files from "initial_dataset/PetImages/Dog" to "Dataset/train"
for filename in os.listdir(dog_source_dir):
    if filename.endswith(".jpg"):
        shutil.copy(os.path.join(dog_source_dir, filename), os.path.join(train_dir, filename))

print("Operation completed: The 'initial_dataset' folder has been renamed, and the files have been copied to 'Dataset/train'.")
