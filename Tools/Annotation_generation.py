import os
from PIL import Image, ImageDraw

# Ajouter un temoin
def annotate_images(source_folder, destination_folder):
    # Crée le dossier de destination s'il n'existe pas
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Parcours les fichiers du dossier source
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)

        try:
            # Ouvre l'image
            image = Image.open(source_path)
            draw = ImageDraw.Draw(image)

            # Vérifie si le nom de l'image contient "cat"
            if "cat" in filename.lower():
                # Dessine un cercle vert
                draw.ellipse([(10, 10), (30, 30)], outline="green", width=2)

            # Vérifie si le nom de l'image contient "dog"
            if "dog" in filename.lower():
                # Dessine un carré rouge
                draw.rectangle([(10, 10), (30, 30)], outline="red", width=2)

            # Enregistre l'image annotée dans le dossier de destination
            image.save(destination_path)
            
            print(f"Image {filename} annotée est enregistrée.")

        except Exception as e:
            print(f"Erreur lors de l'annotation de l'image {filename}: {str(e)}")

annotate_images("Dataset/train", "Dataset_Annotated")
