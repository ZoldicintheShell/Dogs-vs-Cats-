import os
from PIL import Image, ImageFilter

def blur_images(source_folder, destination_folder):
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
            
            # Applique un flou gaussien à l'image
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))  # Vous pouvez ajuster le rayon selon vos besoins
            
            # Enregistre l'image floutée dans le dossier de destination
            blurred_image.save(destination_path)
            
            print(f"Image {filename} floutée est enregistrée.")

        except Exception as e:
            print(f"Erreur lors du floutage de l'image {filename}: {str(e)}")

blur_images("Dataset/train", "Dataset_Blurred")
