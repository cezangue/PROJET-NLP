# PROJET-NLP
Il s'agit ici de faire une analyse des discours politique à partir du Naturel Language Processing. Par la suite nous avons deployer une application sur Streamlit


Ce projet permet d'analyser un discours en extrayant les thèmes principaux, en générant des résumés, et en créant des nuages de mots (Word Clouds). Le discours peut être fourni sous forme de fichier PDF ou via une URL.

## Fonctionnalités

- Extraction de texte à partir d'un fichier PDF ou d'une URL.
- Prétraitement du texte (nettoyage, tokenisation, etc.).
- Génération d'embeddings avec Sentence-BERT.
- Clustering des phrases pour identifier les thèmes principaux.
- Génération de résumés pour chaque thème.
- Extraction de mots-clés avec TF-IDF et RAKE.
- Création de nuages de mots pour chaque thème.

## Installation

1. Clonez ce repository :

   ```bash
   git clone https://github.com/cezangue/PROJET-NLP.git
   cd PROJET-NLP

2. Installez les dépendances nécessaires :

bash
Copy
pip install -r requirements.txt
Téléchargez les ressources nécessaires de NLTK :

bash
Copy
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
Utilisation
Exécutez le script principal :

## Execution
bash
Copy
python main.py
Suivez les instructions à l'écran pour fournir le discours (via un fichier PDF ou une URL).

Exemple de Données
Un exemple de fichier PDF est disponible dans le dossier data/. Vous pouvez l'utiliser pour tester le projet.

Dépendances
Les dépendances sont listées dans le fichier requirements.txt.

## Auteur
1.TAGNE TCHINDA RINEL V. (CMR)
2.BONKOUNGOU EMMANUEL (Burk.)

