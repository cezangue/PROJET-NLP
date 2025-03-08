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
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake

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
        print(f"Erreur lors de la lecture du fichier PDF : {e}")
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
            print("La balise contenant le discours n'a pas été trouvée.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête : {e}")
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
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    phrases = nltk.sent_tokenize(texte, language='french')
    phrases_nettoyees = nettoyer_phrases(phrases)
    
    if len(phrases_nettoyees) < 2:
        print("Erreur : Le texte ne contient pas suffisamment de phrases pour le clustering.")
        return None, None

    embeddings = model.encode(phrases_nettoyees)
    return embeddings, phrases_nettoyees

# Déterminer le nombre optimal de clusters
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
        print(f"Nombre de clusters : {k}, Silhouette : {silhouette_avg}, Inertie : {kmeans.inertia_}")

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
    plt.show()

    meilleur_k = np.argmax(silhouettes) + 2
    print(f"Nombre optimal de thèmes : {meilleur_k}")
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
    choix = input("Choisissez comment fournir le discours (1: Fichier PDF, 2: URL) : ")

    if choix == '1':
        chemin_pdf = input("Entrez le chemin du fichier PDF : ")
        texte_brut = extraire_texte_pdf(chemin_pdf)
    elif choix == '2':
        url = input("Entrez l'URL du discours : ")
        texte_brut = scraper_discours(url)
    else:
        print("Choix invalide.")
        return

    if texte_brut:
        print("Texte brut :")
        print(texte_brut)

        # Prétraitement du texte
        texte_pretraite = pretraiter_texte(texte_brut)

        # Obtenir les embeddings
        embeddings, phrases = obtenir_embeddings(texte_pretraite)

        # Déterminer le nombre optimal de thèmes
        nombre_optimal_themes = trouver_nombre_optimal_themes(embeddings)

        # Extraction des thèmes par clustering
        themes = extraire_themes(embeddings, phrases, nombre_optimal_themes)

        # Afficher les thèmes extraits
        for theme, phrases in themes.items():
            print(f"\n{theme} :")
            print("Phrases associées :")
            for phrase in phrases:
                print(f"- {phrase}")
            mots_cles_tfidf = extraire_mots_cles_tfidf(phrases)
            print(f"Mots clés (TF-IDF) : {[mot for mot, _ in mots_cles_tfidf]}")

if __name__ == "__main__":
    main()
