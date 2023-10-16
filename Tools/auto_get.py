######
# AUTOGET WITH DATA CLEANING
######

import os
import shutil
import tensorflow as tf
import pathlib

# Define a function to download and extract the dataset
def download_and_extract_dataset(dataset_url, destination_dir):
    archive = tf.keras.utils.get_file(fname="./kagglecatsanddogs_5340.zip", origin=dataset_url, extract=True, cache_dir=destination_dir)
    return pathlib.Path(archive).with_suffix('')

# Define a function to remove corrupted images from a directory
def remove_corrupted_images(directory):
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(directory, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)

# Define a function to rename files by adding folder names as prefixes
def rename_files_with_folder_prefix(directory):
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                source_path = os.path.join(folder_path, file)

                if os.path.isfile(source_path):
                    new_name = f"{folder.lower()}.{file.lower()}"
                    destination_path = os.path.join(folder_path, new_name)

                    os.rename(source_path, destination_path)
                    print(f"Renaming of {file} in {new_name}")

# Define a function to organize the dataset into a training directory
def organize_dataset_for_training(initial_dataset_dir, train_dir):
    os.rename(initial_dataset_dir, "initial_dataset")
    os.makedirs(train_dir, exist_ok=True)

    cat_source_dir = "initial_dataset/PetImages/Cat"
    dog_source_dir = "initial_dataset/PetImages/Dog"

    for source_dir in [cat_source_dir, dog_source_dir]:
        for filename in os.listdir(source_dir):
            if filename.endswith(".jpg"):
                shutil.copy(os.path.join(source_dir, filename), os.path.join(train_dir, filename))

    print("Operation completed: The 'initial_dataset' folder has been renamed, and the files have been copied to 'Dataset/train'.")

if __name__ == "__main__":
    # Dataset URL
    dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

    # Get the current directory
    current_directory = pathlib.Path.cwd()

    # Download and extract the dataset
    data_dir = download_and_extract_dataset(dataset_url, current_directory)
    print("The downloaded folder is located at:", data_dir)

    # Remove corrupted images
    remove_corrupted_images(os.path.join("datasets", "PetImages"))
    print("#========== DATASET WELL CLEANED")

    # Rename files by adding folder names as prefixes
    rename_files_with_folder_prefix(os.path.join("datasets", "PetImages"))

    # Organize the dataset for training
    organize_dataset_for_training("datasets", "Dataset/train")

    # Move "test_set" into "Dataset" folder
    shutil.move('Test_set', 'Dataset')
