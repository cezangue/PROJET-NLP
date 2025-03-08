import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
import PyPDF2
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer

# Télécharger les ressources nécessaires de NLTK
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

# Génération des embeddings avec Sentence-BERT
def obtenir_embeddings(texte):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    phrases = nltk.sent_tokenize(texte, language='french')
    embeddings = model.encode(phrases)
    return embeddings, phrases

# Déterminer le nombre optimal de clusters
def trouver_nombre_optimal_themes(embeddings):
    silhouettes = []
    max_k = 10  # Limite pour le nombre de clusters

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
        silhouettes.append(silhouette_avg)

    meilleur_k = np.argmax(silhouettes) + 2
    return meilleur_k

# Clustering avec KMeans pour extraire les thèmes
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

# Extraction des mots-clés avec TF-IDF
def extraire_mots_cles_tfidf(phrases):
    vectorizer = TfidfVectorizer(stop_words=list(mots_a_supprimer))
    X = vectorizer.fit_transform(phrases)
    mots_cles = vectorizer.get_feature_names_out()
    scores = np.asarray(X.sum(axis=0)).flatten()
    mots_cles_avec_scores = [(mot, score) for mot, score in zip(mots_cles, scores) if score > 0]
    mots_cles_avec_scores.sort(key=lambda x: x[1], reverse=True)
    return mots_cles_avec_scores[:10]

# Fonction principale
def main():
    st.title("Analyse de discours")
    texte_brut = None

    choix = st.radio("Choisissez comment fournir le discours :", ("Via un fichier PDF", "Via un lien web"))

    if choix == "Via un fichier PDF":
        uploaded_file = st.file_uploader("Veuillez importer votre fichier PDF :", type=["pdf"])
        if uploaded_file is not None:
            texte_brut = extraire_texte_pdf(uploaded_file)
    else:
        url = st.text_input("Entrez l'URL du discours :")
        if url:
            texte_brut = scraper_discours(url)

    if texte_brut:
        st.write("Texte brut (scrapé ou extrait du PDF) :")
        st.write(texte_brut)

        # Prétraitement du texte
        texte_pretraite = pretraiter_texte(texte_brut)
        st.write("Texte prétraité :")
        st.write(texte_pretraite)

        # Obtenir les embeddings
        embeddings, phrases = obtenir_embeddings(texte_pretraite)

        # Déterminer le nombre optimal de thèmes
        nombre_optimal_themes = trouver_nombre_optimal_themes(embeddings)

        # Extraction des thèmes par clustering
        themes = extraire_themes(embeddings, phrases, nombre_optimal_themes)

        # Afficher les thèmes extraits
        st.sidebar.header("Thèmes identifiés")
        for theme, phrases in themes.items():
            st.sidebar.subheader(theme)
            mots_cles_tfidf = extraire_mots_cles_tfidf(phrases)
            st.sidebar.write(f"Mots clés (TF-IDF) : {[mot for mot, _ in mots_cles_tfidf]}")

            # Génération d'un Word Cloud pour chaque thème
            texte_theme = ' '.join(phrases)
            wordcloud = WordCloud(width=400, height=200, background_color='white').generate(texte_theme)
            st.sidebar.image(wordcloud.to_array(), caption=theme)

if __name__ == "__main__":
    main()
