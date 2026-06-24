# 🏆 World Cup 2026: Monte Carlo Simulation Engine

[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)

## 🎯 Project Overview

Un moteur de simulation haute performance conçu pour prédire les résultats de la Coupe du Monde FIFA 2026. Ce système repose sur une simulation de Monte-Carlo (10 000 itérations) permettant de modéliser l'incertitude du sport et d'obtenir des probabilités de victoire statistiquement significatives pour chaque nation.

### 🌟 Key Features
- **Moteur de Simulation Monte-Carlo**: 10 000 itérations par exécution pour garantir la convergence statistique.
- **Grille Officielle FIFA 2026**: Respect strict du format à 48 équipes (phases de groupes et tableau éliminatoire complet).
- **Intégration de Données Multi-Sources**: Analyse basée sur 5 piliers : Statistiques sportives, données macro-économiques, indicateurs de marché, performance humaine et fair-play.
- **Interface Dashboard**: Visualisation interactive développée avec Streamlit et Plotly.

### 📊 Performance Metrics
- **Convergence Statistique**: 10 000 simulations indépendantes.
- **Temps de Latence**: Calcul complet en temps réel (< 2s).
- **Précision**: Basée sur la pondération multidimensionnelle des forces absolues.

## 📈 Visualizations
Le dashboard inclut :

- Histogrammes dynamiques (Plotly) pour les probabilités de victoire.

- Tableaux de classement des groupes synchronisés.

- Grille symétrique de l'arbre éliminatoire FIFA.

### Prerequisites
- Python 3.10 or higher
- Git for version control

### Step 1: Clone Repository
```bash
git clone [https://github.com/votre-utilisateur/world-cup-predictor.git](https://github.com/votre-utilisateur/world-cup-predictor.git)
cd world-cup-predictor

### Step 2: Install Dependencies
pip install -r requirements.txt

### Step 3: Launch Application
streamlit run src/world_cup_predictor.py

