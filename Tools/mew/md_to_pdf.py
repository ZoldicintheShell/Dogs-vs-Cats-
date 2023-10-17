import markdown2
import re
from weasyprint import HTML, CSS
from PIL import Image
from sys import argv, stderr

# Path arguments are relative paths to the root of mewtwo py projects 
#
# python3 md_to_pdf.py <style_path> <output_path>

def mew_md_to_pdf(input_path, style_path, output_path):
    """
    Convert markdown to pdf with custom styling
    Parameters:
    - input_path: String path to input markdown
    - style_path: String Path to css styling used
    - output_path: String path where to save the pdf
    Returns None
    """
    with open(input_path, "r", encoding="utf-8") as md_file:
        md_content = md_file.read()

    # Convertir le Markdown en HTML
    html_content = markdown2.markdown(md_content)

    # Rechercher les balises d'images Markdown et remplacer par des balises HTML
    html_content_with_images = re.sub(r'!\[.*?\]\((.*?)\)', r'<img src="\1" />', html_content)

    # Charger le contenu du fichier CSS
    with open(style_path, "r", encoding="utf-8") as css_file:
        custom_css = css_file.read()

    # Générer le PDF à partir du HTML avec mise en page personnalisée
    final_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    {custom_css}
    </style>
    </head>
    <body>
    {html_content_with_images}
    </body>
    </html>
    """

    html = HTML(string=final_html, base_url="./")  # L'argument base_url aide à résoudre les chemins relatifs
    pdf_bytes = html.write_pdf()

    # Enregistrer le PDF résultant
    with open(output_path, "wb") as pdf_file:
        pdf_file.write(pdf_bytes)

    print("Conversion terminée.")


def main():
    if (len(argv) != 3):
        print('Arguments error.', file=stderr)
        return 1

    style_path = argv[1]
    output_path = argv[2]
    mew_md_to_pdf('/home/h3x/reading/metaheuristic/metaheuristic.md', 'css/style1.css', './tt2.pdf')


if __name__ == '__main__':
    main()