import os
import shutil
import pandas as pd


def fusionner_fichiers_csv(repertoire_fusion, fichier_sortie):
    """
    Fusionne tous les fichiers CSV dans le répertoire spécifié en un seul fichier CSV de sortie.

    Args:
        repertoire_fusion (str): Le chemin du répertoire contenant les fichiers CSV à fusionner.
        fichier_sortie (str): Le nom du fichier CSV de sortie.

    Returns:
        str: Un message indiquant que la fusion a été effectuée avec succès.
    """
    # Liste pour stocker les DataFrames de chaque fichier CSV
    dataframes = []

    # Parcourir les fichiers CSV dans le répertoire
    for fichier in os.listdir(repertoire_fusion):
        if fichier.endswith(".csv"):
            chemin_fichier = os.path.join(repertoire_fusion, fichier)
            # Lire chaque fichier CSV dans un DataFrame
            df = pd.read_csv(chemin_fichier)
            dataframes.append(df)

    # Fusionner tous les DataFrames en un seul
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Enregistrer le DataFrame fusionné dans un fichier CSV
    merged_df.to_csv(fichier_sortie, index=False)

    return f"Les fichiers CSV ont été fusionnés dans {fichier_sortie}."


#---------
# Définir le répertoire de base où vous souhaitez rechercher les fichiers
repertoire_base = "Experiment1"

# Créer le répertoire "experiments_records" s'il n'existe pas
dossier_cible = "experiments_records"
if not os.path.exists(dossier_cible):
    os.mkdir(dossier_cible)

# Mot clé à rechercher dans le nom des fichiers
mot_cle = "experiment_n°"

# Parcourir le répertoire de base
for root, dirs, files in os.walk(repertoire_base):
    for fichier in files:
        # Vérifier si le nom du fichier contient le mot clé
        if mot_cle in fichier:
            # Construire le chemin complet du fichier source
            chemin_source = os.path.join(root, fichier)
            
            # Construire le chemin complet du fichier cible dans le dossier "experiments_records"
            chemin_cible = os.path.join(dossier_cible, fichier)
            
            # Copier le fichier dans le dossier cible
            shutil.copy(chemin_source, chemin_cible)
            print(f"Fichier copié : {fichier}")

print("La recherche est terminée.")


# Exemple d'utilisation de la fonction
repertoire_source = "experiments_records"
fichier_sortie = "fusion.csv"
message = fusionner_fichiers_csv(repertoire_source, fichier_sortie)
print(message)