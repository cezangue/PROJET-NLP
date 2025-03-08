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
    print("Choisissez comment fournir le discours :")
    print("1. Via un fichier PDF")
    print("2. Via un lien web")
    choix = input("Entrez 1 ou 2 : ")
    return choix

# Fonction pour demander à l'utilisateur comment fournir le fichier PDF
def demander_choix_pdf():
    print("Choisissez comment fournir le fichier PDF :")
    print("1. Entrer le chemin d'accès local")
    print("2. Importer un fichier depuis votre ordinateur")
    choix_pdf = input("Entrez 1 ou 2 : ")
    return choix_pdf

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
            print("La balise contenant le discours n'a pas été trouvée.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête : {e}")
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
        print(f"Erreur lors du prétraitement du texte : {e}")
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
        print(f"Erreur lors du nettoyage des phrases : {e}")
        return None

# Étape 4 : Génération des embeddings avec Sentence-BERT
def obtenir_embeddings(texte):
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')  # Utiliser le CPU
        phrases = nltk.sent_tokenize(texte, language='french')
        phrases_nettoyees = nettoyer_phrases(phrases)
        
        if len(phrases_nettoyees) < 2:
            print("Erreur : Le texte ne contient pas suffisamment de phrases pour le clustering.")
            return None, None
        
        embeddings = model.encode(phrases_nettoyees, device='cpu')  # Assurer que les embeddings sont calculés sur le CPU
        return embeddings, phrases_nettoyees
    except Exception as e:
        print(f"Erreur lors de la génération des embeddings : {e}")
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
    except Exception as e:
        print(f"Erreur lors de la détermination du nombre optimal de clusters : {e}")
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
        print(f"Erreur lors de l'extraction des thèmes : {e}")
        return None

# Extraction des mots clés avec TF-IDF
def extraire_mots_cles_tfidf(phrases):
    try:
        vectorizer = TfidfVectorizer(stop_words=list(mots_a_supprimer))  # Passer la liste de mots à supprimer
        X = vectorizer.fit_transform(phrases)
        mots_cles = vectorizer.get_feature_names_out()
        scores = np.asarray(X.sum(axis=0)).flatten()
        mots_cles_avec_scores = [(mot, score) for mot, score in zip(mots_cles, scores) if score > 0]
        mots_cles_avec_scores.sort(key=lambda x: x[1], reverse=True)
        return mots_cles_avec_scores[:10]  # Retourne les 10 mots clés les plus pertinents
    except Exception as e:
        print(f"Erreur lors de l'extraction des mots-clés avec TF-IDF : {e}")
        return []

# Extraction des mots clés avec RAKE
def extraire_mots_cles_rake(texte):
    try:
        r = Rake()
        r.extract_keywords_from_text(texte)
        return r.get_ranked_phrases()[:10]
    except Exception as e:
        print(f"Erreur lors de l'extraction des mots-clés avec RAKE : {e}")
        return []

# Fonction pour résumer les phrases d'un thème
def resumer_phrases(phrases):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device='cpu')  # Utiliser le CPU
        texte_a_resumer = ' '.join(phrases)
        resum = summarizer(texte_a_resumer, max_length=50, min_length=25, do_sample=False)
        return resum[0]['summary_text']
    except Exception as e:
        print(f"Erreur lors de la génération du résumé : {e}")
        return "Résumé non disponible."

# Fonction pour filtrer les noms et les adjectifs
def filtrer_noms_et_adjectifs(texte):
    try:
        # Tokeniser le texte en mots
        mots = nltk.word_tokenize(texte, language='french')
        
        # Étiqueter les mots avec leur partie du discours
        mots_etiquetes = nltk.pos_tag(mots)
        
        # Filtrer pour garder uniquement les noms et les adjectifs
        mots_filtres = []
        for mot, etiquette in mots_etiquetes:
            # Noms : NN (singular), NNS (plural), NNP (proper noun singular), NNPS (proper noun plural)
            # Adjectifs : JJ (adjective), JJR (comparative), JJS (superlative)
            if etiquette.startswith('NN') or etiquette.startswith('JJ'):
                mots_filtres.append(mot)
        
        # Reconstruire le texte filtré
        texte_filtre = ' '.join(mots_filtres)
        return texte_filtre
    except Exception as e:
        print(f"Erreur lors du filtrage des noms et adjectifs : {e}")
        return texte

# Fonction pour synthétiser le thème et les mots clés (version finale filtrée)
def synthesiser_theme_et_mots_cles(theme, mots_cles_tfidf, mots_cles_rake):
    try:
        # Combiner les mots-clés de TF-IDF et RAKE
        tous_mots_cles = list(set(mots_cles_tfidf + mots_cles_rake))
        
        # Créer une phrase de synthèse
        synthese = f"{theme}: Ce thème aborde des sujets tels que {', '.join(tous_mots_cles[:5])}."
        
        # Filtrer la synthèse pour ne garder que les noms et les adjectifs
        synthese_filtree = filtrer_noms_et_adjectifs(synthese)
        
        return synthese_filtree
    except Exception as e:
        print(f"Erreur lors de la synthèse du thème et des mots-clés : {e}")
        return "Synthèse non disponible."

# Fonction pour créer un tableau de synthèses
def creer_tableau_syntheses(themes):
    try:
        tableau_syntheses = []
        
        for theme, phrases in themes.items():
            # Extraire les mots-clés avec TF-IDF et RAKE
            mots_cles_tfidf = [mot for mot, _ in extraire_mots_cles_tfidf(phrases)]
            mots_cles_rake = extraire_mots_cles_rake(' '.join(phrases))
            
            # Synthèse finale filtrée
            synthese_finale = synthesiser_theme_et_mots_cles(theme, mots_cles_tfidf, mots_cles_rake)
            
            # Ajouter les résultats au tableau
            tableau_syntheses.append({
                "Thème": theme,
                "Synthèse Finale": synthese_finale
            })
        
        return tableau_syntheses
    except Exception as e:
        print(f"Erreur lors de la création du tableau de synthèses : {e}")
        return None

# Fonction pour afficher le tableau de synthèses
def afficher_tableau_syntheses(tableau_syntheses):
    try:
        if tableau_syntheses:
            # Préparer les données pour le tableau
            donnees_tableau = []
            for synthese in tableau_syntheses:
                # Formater chaque cellule pour permettre un retour à la ligne
                synthese_finale = "\n".join(textwrap.wrap(synthese["Synthèse Finale"], width=75))  # Largeur maximale de 50 caractères
                
                ligne = [
                    synthese["Thème"],
                    synthese_finale
                ]
                donnees_tableau.append(ligne)
            
            # En-têtes du tableau
            en_tetes = ["Thème", "Synthèse Finale"]
            
            # Afficher le tableau avec une largeur adaptée
            print(tabulate(donnees_tableau, headers=en_tetes, tablefmt="grid", maxcolwidths=[None, 50]))
        else:
            print("Aucune synthèse disponible pour afficher.")
    except Exception as e:
        print(f"Erreur lors de l'affichage du tableau de synthèses : {e}")

# Étape 7 : Afficher les thèmes extraits avec résumés et mots clés (version finale filtrée)
def afficher_themes(themes):
    try:
        print("\nPrincipaux thèmes identifiés :")
        for theme, phrases in themes.items():
            resume = resumer_phrases(phrases)
            mots_cles_tfidf = [mot for mot, _ in extraire_mots_cles_tfidf(phrases)]
            mots_cles_rake = extraire_mots_cles_rake(' '.join(phrases))
            
            print(f"{theme}: {resume}")
            print(f"Mots clés (TF-IDF): {mots_cles_tfidf}")
            print(f"Mots clés (RAKE): {mots_cles_rake}")

            # Synthèse finale filtrée
            synthese_finale = synthesiser_theme_et_mots_cles(theme, mots_cles_tfidf, mots_cles_rake)
            print(f"Synthèse Finale (noms et adjectifs): {synthese_finale}\n")
    except Exception as e:
        print(f"Erreur lors de l'affichage des thèmes : {e}")

# Étape 8 : Génération des Word Clouds pour les thèmes
def generer_wordcloud(themes):
    try:
        for theme, phrases in themes.items():
            texte_pretraite = ' '.join(phrases)
            if texte_pretraite:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texte_pretraite)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(theme)
                plt.show()
            else:
                print(f"Aucun mot pertinent trouvé pour le {theme}. Impossible de générer le word cloud.")
    except Exception as e:
        print(f"Erreur lors de la génération du Word Cloud : {e}")

# Fonction principale
def main():
    choix = demander_choix()

    if choix == '1':
        choix_pdf = demander_choix_pdf()
        if choix_pdf == '1':
            chemin_pdf = input("Entrez le chemin du fichier PDF : ")
            texte_brut = extraire_texte_pdf(chemin_pdf)
        elif choix_pdf == '2':
            print("Veuillez importer votre fichier PDF :")
            uploaded = files.upload()
            if uploaded:
                fichier_nom = list(uploaded.keys())[0]
                texte_brut = extraire_texte_pdf(fichier_nom)
            else:
                print("Aucun fichier n'a été importé.")
                return
        else:
            print("Choix invalide. Veuillez réessayer.")
            return
    elif choix == '2':
        url = input("Entrez l'URL du discours : ")
        texte_brut = scraper_discours(url)
    else:
        print("Choix invalide. Veuillez réessayer.")
        return

    if texte_brut:
        print("Texte brut (scrapé ou extrait du PDF) :")
        print(texte_brut)
        print("\n" + "=" * 80 + "\n")

        # Prétraitement du texte
        texte_pretraite = pretraiter_texte(texte_brut)
        if texte_pretraite is None:
            print("Erreur : Le texte n'a pas pu être prétraité.")
            return

        # Obtenir les embeddings
        embeddings, phrases = obtenir_embeddings(texte_pretraite)
        if embeddings is None or phrases is None:
            print("Erreur : Impossible de générer les embeddings. Vérifiez le texte d'entrée.")
            return

        # Déterminer le nombre optimal de thèmes
        nombre_optimal_themes = trouver_nombre_optimal_themes(embeddings)
        if nombre_optimal_themes is None:
            print("Erreur : Impossible de déterminer le nombre optimal de thèmes.")
            return

        # Extraction des thèmes par clustering
        themes = extraire_themes(embeddings, phrases, nombre_optimal_themes)
        if themes is None:
            print("Erreur : Impossible d'extraire les thèmes.")
            return

        # Afficher les thèmes extraits avec résumés
        afficher_themes(themes)

        # Générer les Word Clouds pour les thèmes
        generer_wordcloud(themes)

        # Créer et afficher le tableau de synthèses
        tableau_syntheses = creer_tableau_syntheses(themes)
        if tableau_syntheses:
            print("\nTableau des Synthèses :")
            afficher_tableau_syntheses(tableau_syntheses)

    else:
        print("Le scraping ou l'extraction du PDF a échoué. Veuillez vérifier votre entrée.")

# Exécuter le programme
if __name__ == "__main__":
    main()
