"""
Classification Résidence Principale / Résidence Secondaire
==========================================================
Pipeline :
  1. Extraction de features à partir des courbes de charge brutes
  2. Clustering k-means pour labelliser automatiquement RP vs RS
  3. Classification supervisée (régression logistique, MLP)
  4. Évaluation avec matrice de confusion et métriques

Données : courbes de charge Enedis RES2 6-9 kVA, pas 30 min.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, silhouette_score,
    accuracy_score, f1_score
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ── Paramètres ──────────────────────────────────────────────────

STEP_H = 0.5          # pas temporel en heures
NIGHT_START = 2        # début de la plage nocturne (heure)
NIGHT_END = 5          # fin de la plage nocturne
OCCUPATION_FLOOR_WH = 2000  # plancher pour le seuil d'occupation (Wh)
OCCUPATION_RATIO = 0.3      # ratio de la médiane pour le seuil adaptatif


# ── Chargement et nettoyage ─────────────────────────────────────

def charger_donnees(path, col_id="id", col_dt="horodate", col_val="valeur"):
    """Charge le CSV brut et normalise les colonnes."""
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=None, engine="python", encoding="latin-1")

    df.columns = df.columns.str.strip().str.replace('"', "", regex=False).str.lower()
    df = df.rename(columns={col_id: "pdl_id", col_dt: "datetime", col_val: "p_w"})

    # Parsing des dates (gère les formats avec et sans timezone)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    df = df.dropna(subset=["pdl_id", "datetime", "p_w"])
    df["p_w"] = pd.to_numeric(df["p_w"], errors="coerce")
    df = df.dropna(subset=["p_w"])

    # Colonnes temporelles
    df["date"] = df["datetime"].dt.floor("D")
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["hh_index"] = (df["hour"] * 60 + df["datetime"].dt.minute) // 30

    return df


# ── Feature engineering ─────────────────────────────────────────

def _runs_and_gaps(series_bool):
    """Calcule les durées moyennes des séjours et des absences."""
    runs, gaps = [], []
    run_len = gap_len = 0
    for occupied in series_bool:
        if occupied:
            run_len += 1
            if gap_len > 0:
                gaps.append(gap_len)
                gap_len = 0
        else:
            gap_len += 1
            if run_len > 0:
                runs.append(run_len)
                run_len = 0
    if run_len > 0:
        runs.append(run_len)
    if gap_len > 0:
        gaps.append(gap_len)
    return {
        "mean_active_streak": np.mean(runs) if runs else 0,
        "max_gap_len": max(gaps) if gaps else 0,
    }


def _entropie_saisonniere(monthly_energy):
    """Entropie normalisée de la répartition mensuelle de la consommation.
    Proche de 1 → consommation uniforme sur l'année (RP).
    Proche de 0 → concentrée sur quelques mois (RS vacances).
    """
    p = monthly_energy.values.astype(float)
    p = p[p > 0]
    if len(p) < 2:
        return 0.0
    p = p / p.sum()
    h = -np.sum(p * np.log(p))
    return h / np.log(len(p))


def extraire_features(df):
    """Calcule les features discriminantes pour chaque PDL.

    Retourne un DataFrame indexé par pdl_id avec les colonnes :
        - active_day_rate : taux de jours "occupés" (seuil adaptatif)
        - max_gap_len : plus longue absence consécutive (jours)
        - night_active_ratio : ratio de nuits avec talon > 100 W
        - entropy_norm : entropie saisonnière normalisée
        - mean_active_streak : durée moyenne des séjours (jours)
        - cv_daily : coefficient de variation de la conso journalière
    """
    # Agrégation journalière
    daily = (
        df.assign(energy_wh=df["p_w"] * STEP_H)
          .groupby(["pdl_id", "date"], as_index=False)
          .agg(daily_wh=("energy_wh", "sum"), n_steps=("p_w", "size"))
    )
    daily["month"] = daily["date"].dt.month

    # CV journalier par PDL
    stats = daily.groupby("pdl_id")["daily_wh"].agg(["mean", "std", "median"])
    stats["cv_daily"] = stats["std"] / (stats["mean"] + 1)
    stats = stats.rename(columns={"median": "median_wh"})

    # Seuil d'occupation adaptatif : max(2 kWh, 30% de la médiane)
    seuils = np.maximum(OCCUPATION_FLOOR_WH, OCCUPATION_RATIO * stats["median_wh"])
    daily = daily.merge(seuils.rename("seuil"), left_on="pdl_id", right_index=True)
    daily["occupied"] = daily["daily_wh"] >= daily["seuil"]

    # Taux d'occupation
    occ = daily.groupby("pdl_id")["occupied"].mean().rename("active_day_rate")

    # Séjours et absences
    streaks = (
        daily.sort_values(["pdl_id", "date"])
             .groupby("pdl_id")["occupied"]
             .apply(lambda s: pd.Series(_runs_and_gaps(s.values)))
    )
    if isinstance(streaks, pd.Series):
        streaks = streaks.unstack()

    # Talon nocturne
    night = df[df["hour"].between(NIGHT_START, NIGHT_END - 1)]
    night_daily = night.groupby(["pdl_id", "date"])["p_w"].mean()
    night_ratio = night_daily.groupby("pdl_id").apply(
        lambda s: (s > 100).mean()
    ).rename("night_active_ratio")

    # Entropie saisonnière
    monthly = daily.groupby(["pdl_id", "month"])["daily_wh"].mean().unstack(fill_value=0)
    entropy = monthly.apply(_entropie_saisonniere, axis=1).rename("entropy_norm")

    # Assemblage
    features = (
        stats[["cv_daily"]]
        .join(occ).join(streaks).join(night_ratio).join(entropy)
        .reset_index()
    )
    return features


# ── Clustering ──────────────────────────────────────────────────

def clustering_kmeans(features, feature_cols, n_clusters=5, random_state=42):
    """Applique k-means et mappe chaque cluster vers RP (0) ou RS (1).

    Le mapping se fait par la médiane du taux d'occupation :
    les clusters avec un faible taux moyen d'occupation sont étiquetés RS.
    """
    X = features[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=30, random_state=random_state)
    clusters = km.fit_predict(X_scaled)

    # Mapping automatique : les clusters à faible taux d'occupation - RS
    # On utilise la feature "active_day_rate" comme proxy
    cluster_occ = pd.Series(
        features["active_day_rate"].values, index=range(len(features))
    ).groupby(clusters).mean()

    seuil_occ = cluster_occ.median()  # seuil adaptatif
    rs_clusters = set(cluster_occ[cluster_occ < seuil_occ].index)

    labels = np.array([1 if c in rs_clusters else 0 for c in clusters])

    sil = silhouette_score(X_scaled, clusters)

    return labels, clusters, scaler, km, sil


# ── Classification supervisée ──────────────────────────────────

def entrainer_classifieurs(X, y, feature_names=None, random_state=42):
    """Entraîne et évalue trois classifieurs : régression logistique,
    random forest et MLP.

    Utilise une validation croisée stratifiée à 5 plis.
    Retourne un dict {nom: {model, y_pred, rapport, matrice_confusion, ...}}.
    Le random forest fournit en plus les importances de features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modeles = {
        "Régression logistique": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=6, random_state=random_state
        ),
        "Réseau de neurones (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500,
            random_state=random_state, early_stopping=True,
            validation_fraction=0.15
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    resultats = {}

    for nom, model in modeles.items():
        y_pred = cross_val_predict(model, X_scaled, y, cv=cv)

        # Ré-entraîner sur tout le dataset pour avoir un modèle final
        model.fit(X_scaled, y)

        res = {
            "model": model,
            "scaler": scaler,
            "y_pred_cv": y_pred,
            "accuracy": accuracy_score(y, y_pred),
            "f1_rs": f1_score(y, y_pred, pos_label=1),
            "rapport": classification_report(y, y_pred, target_names=["RP", "RS"]),
            "matrice": confusion_matrix(y, y_pred),
        }

        # Le random forest donne les importances de chaque feature
        if hasattr(model, "feature_importances_") and feature_names is not None:
            importances = dict(zip(feature_names, model.feature_importances_))
            res["feature_importances"] = dict(
                sorted(importances.items(), key=lambda x: x[1], reverse=True)
            )

        resultats[nom] = res

    return resultats


def evaluer_sur_dataset_equilibre(X, y, feature_names=None, random_state=42):
    """Évalue sur un dataset équilibré (sous-échantillonnage de la majorité).

    Le prof demande explicitement d'avoir autant de RP que de RS dans le test.
    """
    rp_idx = np.where(y == 0)[0]
    rs_idx = np.where(y == 1)[0]
    n_min = min(len(rp_idx), len(rs_idx))

    rng = np.random.RandomState(random_state)
    rp_sample = rng.choice(rp_idx, size=n_min, replace=False)
    idx_bal = np.concatenate([rp_sample, rs_idx])

    X_bal, y_bal = X[idx_bal], y[idx_bal]
    return entrainer_classifieurs(X_bal, y_bal, feature_names, random_state)


# ── Comparaison avec les labels de référence ───────────────────

def comparer_avec_reference(labels_pred, labels_ref):
    """Compare nos labels avec ceux du prof et affiche les métriques."""
    rapport = classification_report(
        labels_ref, labels_pred, target_names=["RP", "RS"], output_dict=True
    )
    matrice = confusion_matrix(labels_ref, labels_pred)
    return rapport, matrice


# ── Features utilisées pour le clustering et la classification ──

FEATURE_COLS = [
    "active_day_rate",
    "max_gap_len",
    "night_active_ratio",
    "entropy_norm",
    "mean_active_streak",
    "cv_daily",
]


# ── Point d'entrée ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "datas/courbes-de-charges-fictives-res2-6-9.csv"
    ref_path = sys.argv[2] if len(sys.argv) > 2 else "datas/RES2-6-9-labels.csv"

    # 1. Chargement
    print("Chargement des données...")
    df = charger_donnees(data_path)
    print(f"  {len(df):,} mesures, {df['pdl_id'].nunique()} PDL")

    # 2. Features
    print("Extraction des features...")
    features = extraire_features(df)
    print(f"  {len(features)} PDL, {len(FEATURE_COLS)} features")

    # 3. Clustering
    print("\nClustering k-means...")
    labels_km, clusters, scaler_km, km, sil = clustering_kmeans(
        features, FEATURE_COLS, n_clusters=5
    )
    n_rp = (labels_km == 0).sum()
    n_rs = (labels_km == 1).sum()
    print(f"  Silhouette: {sil:.3f}")
    print(f"  RP: {n_rp}, RS: {n_rs}")

    # 4. Comparaison avec le corrigé du prof
    ref = pd.read_csv(ref_path)
    ref.columns = ref.columns.str.strip()
    merged = features.merge(ref, left_on="pdl_id", right_on="id", how="inner")

    if len(merged) > 0:
        y_ref = merged["label"].values
        y_km = labels_km[features["pdl_id"].isin(merged["pdl_id"])]

        print("\n── Clustering vs référence ──")
        rapport_km, mat_km = comparer_avec_reference(y_km, y_ref)
        print(f"  Accuracy: {rapport_km['accuracy']:.3f}")
        print(f"  F1 RS:    {rapport_km['RS']['f1-score']:.3f}")
        print(f"  Matrice:\n{mat_km}")

    # 5. Classification supervisée
    X = features[FEATURE_COLS].values
    y = labels_km  # on utilise les labels k-means comme cible

    print("\n── Classification supervisée (5-fold CV) ──")
    resultats = entrainer_classifieurs(X, y, feature_names=FEATURE_COLS)
    for nom, res in resultats.items():
        print(f"\n{nom}:")
        print(f"  Accuracy: {res['accuracy']:.3f}")
        print(f"  F1 RS:    {res['f1_rs']:.3f}")
        print(res["rapport"])
        if "feature_importances" in res:
            print("  Importances des features:")
            for feat, imp in res["feature_importances"].items():
                print(f"    {feat:25s} {imp:.3f}")

    # 6. Évaluation équilibrée
    print("\n── Évaluation sur dataset équilibré ──")
    resultats_eq = evaluer_sur_dataset_equilibre(X, y, feature_names=FEATURE_COLS)
    for nom, res in resultats_eq.items():
        print(f"\n{nom} (équilibré):")
        print(f"  Accuracy: {res['accuracy']:.3f}")
        print(f"  F1 RS:    {res['f1_rs']:.3f}")
        print(res["rapport"])