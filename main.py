import requests
from bs4 import BeautifulSoup
import re
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import PyPDF2
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
from tabulate import tabulate
import textwrap
import streamlit as st

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Liste des mots à supprimer
mots_a_supprimer = set(nltk.corpus.stopwords.words('french') + [
    'ainsi', 'aussi', 'plus', 'déjà', 'cest', 'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',
    'le', 'la', 'les', 'un', 'une', 'des', 'ce', 'cette', 'ces', 'il', 'elle', 'ils', 'elles',
    'me', 'te', 'nous', 'vous', 'lui', 'leur', 'a', 'à', 'de', 'du', 'dans', 'pour', 'par', 'sur', 'avec', 'sans'
])

# Fonction pour demander à l'utilisateur de choisir une option
def demander_choix():
    choix = st.radio("Choisissez comment fournir le discours :", ("Via un fichier PDF", "Via un lien web"))
    return 1 if choix == "Via un fichier PDF" else 2

# Fonction pour extraire le texte d'un fichier PDF
def extraire_texte_pdf(chemin_pdf):
    try:
        with open(chemin_pdf, 'rb') as fichier:
            lecteur = PyPDF2.PdfReader(fichier)
            texte = ""
            for page in lecteur.pages:
                texte += page.extract_text()
            return texte
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier PDF : {e}")
        return None

# Étape 1 : Scraping du discours depuis un lien
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

# Étape 2 : Prétraitement du texte
def pretraiter_texte(texte):
    try:
        texte = re.sub(r'\n+', ' ', texte)
        texte = re.sub(r'\s+', ' ', texte)
        texte = texte.lower()
        texte = re.sub(r'[^\w\s.]', '', texte)
        return texte
    except Exception as e:
        st.error(f"Erreur lors du prétraitement du texte : {e}")
        return None

# Étape 3 : Nettoyage des phrases
def nettoyer_phrases(phrases):
    try:
        phrases_nettoyees = []
        for phrase in phrases:
            mots = phrase.split()
            mots_nettoyes = [mot for mot in mots if mot not in mots_a_supprimer]
            phrases_nettoyees.append(' '.join(mots_nettoyes))
        return phrases_nettoyees
    except Exception as e:
        st.error(f"Erreur lors du nettoyage des phrases : {e}")
        return None

# Étape 4 : Génération des embeddings avec Sentence-BERT
def obtenir_embeddings(texte):
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
        phrases = nltk.sent_tokenize(texte, language='french')
        phrases_nettoyees = nettoyer_phrases(phrases)
        
        if len(phrases_nettoyees) < 2:
            st.error("Erreur : Le texte ne contient pas suffisamment de phrases pour le clustering.")
            return None, None
        
        embeddings = model.encode(phrases_nettoyees, device='cpu')
        return embeddings, phrases_nettoyees
    except Exception as e:
        st.error(f"Erreur lors de la génération des embeddings : {e}")
        return None, None

# Étape 5 : Déterminer le nombre optimal de clusters avec la méthode du coude
def trouver_nombre_optimal_themes(embeddings):
    try:
        silhouettes = []
        inerties = []
        max_k = 20

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            inerties.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
            silhouettes.append(silhouette_avg)
            st.write(f"Nombre de clusters : {k}, Silhouette : {silhouette_avg}, Inertie : {kmeans.inertia_}")

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
        st.pyplot(plt)  # Afficher la figure dans Streamlit

        meilleur_k = np.argmax(silhouettes) + 2
        st.write(f"Nombre optimal de thèmes : {meilleur_k}")
        return meilleur_k
    except Exception as e:
        st.error(f"Erreur lors de la détermination du nombre optimal de clusters : {e}")
        return None

# Étape 6 : Clustering avec KMeans
def extraire_themes(embeddings, phrases, n_themes):
    try:
        kmeans = KMeans(n_clusters=n_themes, random_state=42)
        kmeans.fit(embeddings)

        themes = {}
        for i, label in enumerate(kmeans.labels_):
            theme_key = f'Thème {label + 1}'
            if theme_key not in themes:
                themes[theme_key] = []
            themes[theme_key].append(phrases[i])

        return themes
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des thèmes : {e}")
        return None

# Fonction pour extraire les mots clés avec TF-IDF
def extraire_mots_cles_tfidf(phrases):
    try:
        vectorizer = TfidfVectorizer(stop_words=list(mots_a_supprimer))
        X = vectorizer.fit_transform(phrases)
        mots_cles = vectorizer.get_feature_names_out()
        scores = np.asarray(X.sum(axis=0)).flatten()
        mots_cles_avec_scores = [(mot, score) for mot, score in zip(mots_cles, scores) if score > 0]
        mots_cles_avec_scores.sort(key=lambda x: x[1], reverse=True)
        return mots_cles_avec_scores[:10]
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des mots-clés avec TF-IDF : {e}")
        return []

# Extraction des mots clés avec RAKE
def extraire_mots_cles_rake(texte):
    try:
        r = Rake()
        r.extract_keywords_from_text(texte)
        return r.get_ranked_phrases()[:10]
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des mots-clés avec RAKE : {e}")
        return []

# Fonction pour résumer les phrases d'un thème
def resumer_phrases(phrases):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device='cpu')
        texte_a_resumer = ' '.join(phrases)
        resum = summarizer(texte_a_resumer, max_length=50, min_length=25, do_sample=False)
        return resum[0]['summary_text']
    except Exception as e:
        st.error(f"Erreur lors de la génération du résumé : {e}")
        return "Résumé non disponible."

# Fonction pour filtrer les noms et les adjectifs
def filtrer_noms_et_adjectifs(texte):
    try:
        mots = nltk.word_tokenize(texte, language='french')
        mots_etiquetes = nltk.pos_tag(mots)
        mots_filtres = [mot for mot, etiquette in mots_etiquetes if etiquette.startswith('NN') or etiquette.startswith('JJ')]
        texte_filtre = ' '.join(mots_filtres)
        return texte_filtre
    except Exception as e:
        st.error(f"Erreur lors du filtrage des noms et adjectifs : {e}")
        return texte

# Fonction pour synthétiser le thème et les mots clés
def synthesiser_theme_et_mots_cles(theme, mots_cles_tfidf, mots_cles_rake):
    try:
        tous_mots_cles = list(set(mots_cles_tfidf + mots_cles_rake))
        synthese = f"{theme}: Ce thème aborde des sujets tels que {', '.join(tous_mots_cles[:5])}."
        synthese_filtree = filtrer_noms_et_adjectifs(synthese)
        return synthese_filtree
    except Exception as e:
        st.error(f"Erreur lors de la synthèse du thème et des mots-clés : {e}")
        return "Synthèse non disponible."

# Fonction principale
def main():
    choix = demander_choix()

    if choix == 1:
        choix_pdf = st.radio("Choisissez comment fournir le fichier PDF :", ("Entrer le chemin d'accès local", "Importer un fichier depuis votre ordinateur"))
        if choix_pdf == 'Entrer le chemin d\'accès local':
            chemin_pdf = st.text_input("Entrez le chemin du fichier PDF :")
            if chemin_pdf:
                texte_brut = extraire_texte_pdf(chemin_pdf)
        elif choix_pdf == 'Importer un fichier depuis votre ordinateur':
            uploaded_file = st.file_uploader("Veuillez importer votre fichier PDF :", type=["pdf"])
            if uploaded_file is not None:
                texte_brut = extraire_texte_pdf(uploaded_file)
            else:
                st.warning("Aucun fichier n'a été importé.")
                return
        else:
            st.warning("Choix invalide. Veuillez réessayer.")
            return
    elif choix == 2:
        url = st.text_input("Entrez l'URL du discours :")
        if url:
            texte_brut = scraper_discours(url)
    else:
        st.warning("Choix invalide. Veuillez réessayer.")
        return

    if texte_brut:
        st.write("Texte brut (scrapé ou extrait du PDF) :")
        st.write(texte_brut)
        st.write("\n" + "=" * 80 + "\n")

        texte_pretraite = pretraiter_texte(texte_brut)
        if texte_pretraite is None:
            st.error("Erreur : Le texte n'a pas pu être prétraité.")
            return

        embeddings, phrases = obtenir_embeddings(texte_pretraite)
        if embeddings is None or phrases is None:
            st.error("Erreur : Impossible de générer les embeddings.")
            return

        nombre_optimal_themes = trouver_nombre_optimal_themes(embeddings)
        if nombre_optimal_themes is None:
            st.error("Erreur : Impossible de déterminer le nombre optimal de thèmes.")
            return

        themes = extraire_themes(embeddings, phrases, nombre_optimal_themes)
        if themes is None:
            st.error("Erreur : Impossible d'extraire les thèmes.")
            return

        afficher_themes(themes)
        generer_wordcloud(themes)

# Exécuter le programme
if __name__ == "__main__":
    main()
