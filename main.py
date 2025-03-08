import requests
from bs4 import BeautifulSoup
import re
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import PyPDF2
from rake_nltk import Rake
import streamlit as st

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Liste des mots à supprimer
mots_a_supprimer = set(nltk.corpus.stopwords.words('french') + [
    'ainsi', 'aussi', 'plus', 'déjà', 'cest', 'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',
    'le', 'la', 'les', 'un', 'une', 'des', 'ce', 'cette', 'ces', 'il', 'elle', 'ils', 'elles',
    'me', 'te', 'nous', 'vous', 'lui', 'leur', 'a', 'à', 'de', 'du', 'dans', 'pour', 'par', 'sur', 'avec', 'sans'
])

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

# Nettoyage des phrases
def nettoyer_phrases(phrases):
    phrases_nettoyees = []
    for phrase in phrases:
        mots = phrase.split()
        mots_nettoyes = [mot for mot in mots if mot not in mots_a_supprimer]
        phrases_nettoyees.append(' '.join(mots_nettoyes))
    return phrases_nettoyees

# Génération des embeddings avec Sentence-BERT
def obtenir_embeddings(texte):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    phrases = nltk.sent_tokenize(texte, language='french')
    phrases_nettoyees = nettoyer_phrases(phrases)
    
    if len(phrases_nettoyees) < 2:
        st.error("Erreur : Le texte ne contient pas suffisamment de phrases pour le clustering.")
        return None, None
    
    embeddings = model.encode(phrases_nettoyees, device='cpu')
    return embeddings, phrases_nettoyees

# Déterminer le nombre optimal de clusters avec la méthode du coude
def trouver_nombre_optimal_themes(embeddings):
    silhouettes = []
    inerties = []
    max_k = 20

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        inerties.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
        silhouettes.append(silhouette_avg)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inerties, marker='o')
    plt.title('Méthode du Coude')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouettes, marker='o')
    plt.title('Coefficient de Silhouette')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Silhouette')
    st.pyplot(plt)

    meilleur_k = np.argmax(silhouettes) + 2
    st.write(f"Nombre optimal de thèmes : {meilleur_k}")
    return meilleur_k

# Clustering avec KMeans
def extraire_themes(embeddings, phrases, n_themes):
    kmeans = KMeans(n_clusters=n_themes, random_state=42)
    kmeans.fit(embeddings)

    themes = {}
    for i, label in enumerate(kmeans.labels_):
        theme_key = f'Thème {label + 1}'
        if theme_key not in themes:
            themes[theme_key] = []
        themes[theme_key].append(phrases[i])

    return themes

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

        texte_pretraite = pretraiter_texte(texte_brut)
        embeddings, phrases = obtenir_embeddings(texte_pretraite)

        if embeddings is not None and phrases is not None:
            nombre_optimal_themes = trouver_nombre_optimal_themes(embeddings)
            if nombre_optimal_themes:
                themes = extraire_themes(embeddings, phrases, nombre_optimal_themes)
                st.write("Thèmes identifiés :")
                for theme, phrases in themes.items():
                    st.write(f"{theme}: {', '.join(phrases[:3])}...")  # Affiche les 3 phrases pour chaque thème

if __name__ == "__main__":
    main()
