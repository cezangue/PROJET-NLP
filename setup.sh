#!/bin/bash

# Mettre à jour pip
pip install --upgrade pip

# Installer Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Ajouter Rust au PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Installer les dépendances
pip install -r requirements.txt

# Démarrer l'application
streamlit run app.py
