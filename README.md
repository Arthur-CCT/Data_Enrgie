# Data & Énergie — Classification, Prévision et Génération de courbes de charge

Projet réalisé dans le cadre de l'UE Data & Énergie. L'objectif est d'appliquer plusieurs algorithmes sur des données de consommation électrique résidentielle issues de l'open data Enedis (segment RES2, 6-9 kVA, pas de 30 minutes sur 1 an).

Le livrable est un dashboard Streamlit qui présente trois fonctionnalités :

1. **Classification RP / RS** : identification des résidences principales et secondaires via clustering (k-means) puis classification supervisée (régression logistique, random forest, MLP)
2. **Prévision** : prédiction de la courbe de charge à J+1 par CNN
3. **Génération** : génération de courbes synthétiques conditionnées au type de résidence via auto-encodeur variationnel (VAE)

## Installation

```bash
git clone <url-du-repo>
cd projet-data-energie
pip install -r requirements.txt
```

Dépendances principales :
- pandas, numpy
- scikit-learn
- torch
- plotly
- streamlit

## Lancer le dashboard

```bash
streamlit run app.py
```

Les données brutes (`courbes-de-charges-fictives-res2-6-9.csv`) et les labels de référence (`RES2-6-9-labels.csv`) sont attendus dans le dossier `datas/`.

## Structure du projet

```
├── app.py                   # Point d'entrée Streamlit (navigation)
├── models/                  # Modules ML
│   ├── classification.py    # Pipeline clustering + classifieurs
│   └── prevision.py         # Pipeline de prévision : agrégation + normalisation + fenêtrage  
├── views/                   # Pages du dashboard
│   ├── classification_view.py
│   ├── prevision_view.py
│   └── generation_view.py
├── notebooks/               # Explorations et brouillons
├── datas/                   # Données (non versionnées)
└── README.md
```

Chaque fonctionnalité est découplée : le modèle ML est dans `models/`, la page Streamlit correspondante dans `views/`. Pour ajouter une fonctionnalité, il suffit de créer le module dans `models/` et de compléter la vue associée.

## Données

Les valeurs du CSV sont des puissances en watts au pas de 30 minutes. Les clients ont une puissance souscrite entre 6 et 9 kVA.

