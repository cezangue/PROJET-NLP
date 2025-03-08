import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
import re
import nltk

# Vérifier si les ressources nécessaires de NLTK sont téléchargées
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Liste des mots à supprimer
mots_a_supprimer = set(nltk.corpus.stopwords.words('french'))

# Fonction pour extraire le texte d'un fichier PDF
def extraire_texte_pdf(uploaded_file):
    try:
        lecteur = PyPDF2.PdfReader(uploaded_file)
        texte = ""
        for page in lecteur.pages:
            texte += page.extract_text()
        return texte
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier PDF : {e}")
        return None

# Scraping du discours depuis un lien
def scraper_discours(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        discours_content = soup.find('div', class_='entry-content')
        if discours_content:
            texte_du_discours = discours_content.get_text(separator="\n")
            return texte_du_discours
        else:
            st.error("La balise contenant le discours n'a pas été trouvée.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la requête : {e}")
        return None

# Prétraitement du texte
def pretraiter_texte(texte):
    texte = re.sub(r'\n+', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte)
    texte = texte.lower()
    texte = re.sub(r'[^\w\s.]', '', texte)
    return texte

# Fonction principale
def main():
    st.title("Analyse de discours")

    choix = st.radio("Choisissez comment fournir le discours :", ("Via un fichier PDF", "Via un lien web"))

    if choix == "Via un fichier PDF":
        uploaded_file = st.file_uploader("Veuillez importer votre fichier PDF :", type=["pdf"])
        if uploaded_file is not None:
            texte_brut = extraire_texte_pdf(uploaded_file)
    else:
        url = st.text_input("Entrez l'URL du discours :")
        if url:
            texte_brut = scraper_discours(url)
        else:
            texte_brut = None

    if texte_brut:
        st.write("Texte brut (scrapé ou extrait du PDF) :")
        st.write(texte_brut)

        # Prétraitement du texte
        texte_pretraite = pretraiter_texte(texte_brut)
        st.write("Texte prétraité :")
        st.write(texte_pretraite)

if __name__ == "__main__":
    main()
