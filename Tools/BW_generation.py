import os
from PIL import Image

def convert_images_to_bw(source_folder, destination_folder):
    # Crée le dossier de destination s'il n'existe pas
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Parcours les fichiers du dossier source
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)

        try:
            # Ouvre l'image en couleur
            image = Image.open(source_path)
            
            # Convertit l'image en noir et blanc
            bw_image = image.convert("L")  # "L" signifie noir et blanc (mode de couleur)
            
            # Enregistre l'image convertie dans le dossier de destination
            bw_image.save(destination_path)
            
            print(f"Image {filename} convertie en noir et blanc est enregistrée.")

        except Exception as e:
            print(f"Erreur lors de la conversion de l'image {filename}: {str(e)}")

convert_images_to_bw("Dataset/train","Dataset_BW")