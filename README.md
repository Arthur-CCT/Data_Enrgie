# Data & Énergie - Classification, Prévision et Génération de courbes de charge

Projet réalisé dans le cadre de l'UE Data & Énergie à l'ENPC (2025). L'objectif est d'appliquer plusieurs méthodes d'apprentissage sur des données de consommation électrique résidentielle issues de l'open data Enedis (segment RES2, 6-9 kVA, pas de 30 minutes sur 1 an).

Le livrable est un dashboard Streamlit qui présente trois fonctionnalités.

## Fonctionnalités

**1. Classification RP / RS**

Identification des résidences principales et secondaires en deux étapes. Un clustering (k-means) labellise automatiquement les clients à partir de features extraites des courbes de charge (taux d'occupation, durée d'absence maximale, talon nocturne, variabilité, entropie saisonnière, ratio weekend/semaine). Trois classifieurs supervisés (régression logistique, random forest, MLP) sont ensuite entraînés sur ces labels et évalués sur un dataset équilibré. Le dashboard affiche les distributions des features, la composition des clusters, les matrices de confusion et les importances du random forest.

**2. Prévision de consommation (CNN-LSTM)**

Prédiction de la consommation de la semaine à venir par décomposition en deux quantités : le total hebdomadaire (prévisible, prédit par un CNN-LSTM à apprentissage résiduel) et la forme jour-de-semaine (habitude stable du foyer, estimée sur l'historique). La prédiction journalière est le produit des deux. La consommation brute est agrégée en énergie quotidienne (kWh), normalisée par foyer, et enrichie de features temporelles (sin/cos du jour de l'année). Le split est chronologique par foyer. Le dashboard compare le modèle à deux baselines naïves (persistence et moyenne pondérée) sur le total hebdomadaire et sur le détail journalier.

**3. Génération de courbes (VAE conditionnel)**

Génération de profils annuels synthétiques conditionnés au type de résidence (RP ou RS) via un auto-encodeur variationnel. Deux architectures sont comparées : un modèle linéaire (couches denses) et un modèle Conv-Attention (convolutions 1D + Transformer). Les convolutions captent les motifs hebdomadaires, le Transformer capte la saisonnalité. Le label est concaténé à l'entrée de l'encodeur et du décodeur. Le dashboard permet de comparer visuellement et statistiquement les courbes générées aux courbes réelles, et de tirer de nouvelles courbes à la demande.

## Installation

```bash
pip install -r requirements.txt
```

Dépendances : pandas, numpy, scikit-learn, torch, plotly, streamlit.

## Lancement

```bash
streamlit run app.py
```

## Données

Les fichiers de données ne sont pas versionnés dans le dépôt (taille et confidentialité). Avant de lancer le dashboard, placer dans le dossier `datas/` les deux fichiers suivants, fournis séparément :

```
datas/courbes-de-charges-fictives-res2-6-9.csv   # courbes de charge
datas/RES2-6-9-labels.csv                        # labels RP / RS
```

Les valeurs du CSV sont des puissances en watts au pas de 30 minutes. Les clients ont une puissance souscrite entre 6 et 9 kVA. Les labels de référence associent chaque identifiant client à un type de résidence (RP ou RS).

## Structure du projet

```
├── app.py                     # Point d'entrée Streamlit (navigation)
├── models/
│   ├── classification.py      # Clustering + classifieurs supervisés
│   ├── prevision.py           # Décomposition niveau × forme, CNN-LSTM résiduel
│   └── generation.py          # CVAE linéaire et Conv-Attention
├── views/
│   ├── classification_view.py
│   ├── prevision_view.py
│   └── generation_view.py
├── datas/                     # Données
└── README.md
```

Chaque fonctionnalité est découplée : le module ML est dans `models/`, la page Streamlit correspondante dans `views/`.

## Auteurs

Hugo Barrat, Arthur Claveau, Baptiste Taret.