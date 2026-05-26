"""
Vue Classification RP / RS
===========================
Présentation du pipeline complet : données, features, clustering,
classification supervisée. Chaque étape est expliquée et justifiée.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.classification import (
    charger_donnees, extraire_features, clustering_kmeans,
    analyse_choix_k, exemples_courbes,
    entrainer_classifieurs, evaluer_sur_dataset_equilibre,
    comparer_avec_reference, FEATURE_COLS,
)

# ── Palette ─────────────────────────────────────────────────────

C = {
    "rp": "#2563eb",
    "rs": "#dc2626",
    "gris": "#6b7280",
    "accent": "#0ea5e9",
    "vert": "#16a34a",
}

NOMS = {
    "active_day_rate": "Taux d'occupation",
    "max_gap_len": "Plus longue absence (j)",
    "night_active_ratio": "Ratio nuits actives",
    "entropy_norm": "Entropie saisonnière",
    "mean_active_streak": "Séjour moyen (j)",
    "cv_daily": "CV conso journalière",
}

DESCRIPTIONS_FEATURES = {
    "active_day_rate": (
        "Proportion de jours où la consommation dépasse un seuil adaptatif "
        "(max entre 2 kWh et 30 % de la médiane du PDL). "
        "Un taux proche de 1 indique une occupation quasi permanente (RP), "
        "un taux bas traduit des absences fréquentes (RS)."
    ),
    "max_gap_len": (
        "Durée en jours de la plus longue période d'absence consécutive. "
        "C'est la feature la plus discriminante : une RP dépasse rarement "
        "2-3 semaines d'absence, alors qu'une RS peut rester vide plusieurs mois."
    ),
    "night_active_ratio": (
        "Proportion de nuits où le talon nocturne (2h-5h) dépasse 100 W. "
        "Un logement occupé maintient un talon permanent (réfrigérateur, "
        "veille, chauffage). Une RS éteinte la nuit descend sous ce seuil."
    ),
    "entropy_norm": (
        "Entropie normalisée de la répartition mensuelle de consommation. "
        "Proche de 1 si la conso est uniforme sur l'année (RP). "
        "Proche de 0 si concentrée sur quelques mois (RS vacances d'été par ex.)."
    ),
    "mean_active_streak": (
        "Durée moyenne des séjours continus (en jours). "
        "Une RP a des séjours très longs (l'occupant y vit), "
        "une RS a des séjours courts et entrecoupés d'absences."
    ),
    "cv_daily": (
        "Coefficient de variation de l'énergie journalière. "
        "Mesure l'irrégularité de la consommation au fil des jours. "
        "Plus volatile pour une RS (alternance occupation/vide) que pour une RP."
    ),
}


# ── Cache ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement des données...")
def charger_et_traiter(data_path):
    df = charger_donnees(data_path)
    features = extraire_features(df)
    return df, features


@st.cache_data(show_spinner="Clustering en cours...")
def lancer_clustering(_features, feature_cols, k):
    return clustering_kmeans(_features, feature_cols, n_clusters=k)


@st.cache_data(show_spinner="Analyse du choix de k...")
def analyser_k(_features, feature_cols):
    return analyse_choix_k(_features, feature_cols)


@st.cache_data(show_spinner="Entraînement des classifieurs...")
def lancer_classification(_X, _y, feature_names):
    res = entrainer_classifieurs(_X, _y, feature_names=feature_names)
    res_eq = evaluer_sur_dataset_equilibre(_X, _y, feature_names=feature_names)
    return res, res_eq


# ── Graphiques ──────────────────────────────────────────────────

def fig_courbes_exemple(exemples, label, couleur, semaines=2):
    """Courbes brutes superposées pour un type de résidence."""
    fig = go.Figure()
    for pdl, data in exemples[label].items():
        # Limiter aux n premières semaines pour la lisibilité
        date_min = data["datetime"].min()
        date_max = date_min + pd.Timedelta(weeks=semaines)
        sub = data[(data["datetime"] >= date_min) & (data["datetime"] < date_max)]
        fig.add_trace(go.Scatter(
            x=sub["datetime"], y=sub["p_w"],
            mode="lines", name=f"PDL ...{str(pdl)[-6:]}",
            line=dict(width=1), opacity=0.8,
        ))
    fig.update_layout(
        yaxis_title="Puissance (W)",
        xaxis_title="",
        height=300, margin=dict(l=50, r=20, t=10, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def fig_elbow_silhouette(df_k):
    """Double axe : inertie (coude) et silhouette."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df_k["k"], y=df_k["inertie"], name="Inertie",
        line=dict(color=C["gris"], width=2), mode="lines+markers",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df_k["k"], y=df_k["silhouette"], name="Silhouette",
        line=dict(color=C["accent"], width=2), mode="lines+markers",
    ), secondary_y=True)
    fig.update_layout(
        xaxis_title="Nombre de clusters (k)",
        height=350, margin=dict(l=50, r=50, t=20, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_yaxes(title_text="Inertie", secondary_y=False)
    fig.update_yaxes(title_text="Score silhouette", secondary_y=True)
    return fig


def fig_feature_detail(features, labels, col):
    """Histogramme d'une feature avec séparation RP/RS."""
    fig = go.Figure()
    for lab, nom, couleur in [(0, "RP", C["rp"]), (1, "RS", C["rs"])]:
        mask = labels == lab
        fig.add_trace(go.Histogram(
            x=features.loc[mask, col], name=nom,
            marker_color=couleur, opacity=0.65, nbinsx=35,
        ))
    fig.update_layout(
        barmode="overlay", height=250,
        xaxis_title=NOMS[col], yaxis_title="Nombre de PDL",
        margin=dict(l=50, r=20, t=10, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def fig_matrice_confusion(matrice, titre=""):
    labels = ["RP", "RS"]
    total = matrice.sum()
    texte = [[f"{v}<br>({v/total:.1%})" for v in row] for row in matrice]
    fig = go.Figure(data=go.Heatmap(
        z=matrice, x=labels, y=labels,
        texttemplate="%{text}", text=texte,
        colorscale=[[0, "#e0f2fe"], [1, "#1e40af"]],
        showscale=False,
    ))
    fig.update_layout(
        title=titre, title_font_size=14,
        xaxis_title="Prédiction", yaxis_title="Réalité",
        width=340, height=340,
        margin=dict(l=60, r=20, t=50, b=60),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def fig_comparaison_modeles(resultats):
    noms = list(resultats.keys())
    acc = [resultats[n]["accuracy"] for n in noms]
    f1 = [resultats[n]["f1_rs"] for n in noms]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy", x=noms, y=acc,
        marker_color=C["rp"], text=[f"{v:.1%}" for v in acc],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="F1 (RS)", x=noms, y=f1,
        marker_color=C["rs"], text=[f"{v:.1%}" for v in f1],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group", yaxis_range=[0, 1.15],
        yaxis_title="Score", legend=dict(orientation="h", y=1.12),
        margin=dict(l=40, r=20, t=50, b=40), height=370,
    )
    return fig


def fig_importances(importances_dict):
    feats = list(importances_dict.keys())
    vals = list(importances_dict.values())
    labels = [NOMS.get(f, f) for f in feats]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=C["accent"],
        text=[f"{v:.1%}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Importance relative",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=170, r=50, t=10, b=40), height=270,
    )
    return fig


def fig_scatter_clusters(features, labels):
    fig = go.Figure()
    for lab, nom, couleur in [(0, "RP", C["rp"]), (1, "RS", C["rs"])]:
        mask = labels == lab
        fig.add_trace(go.Scatter(
            x=features.loc[mask, "active_day_rate"],
            y=features.loc[mask, "max_gap_len"],
            mode="markers", name=nom,
            marker=dict(color=couleur, size=6, opacity=0.7),
        ))
    fig.update_layout(
        xaxis_title=NOMS["active_day_rate"],
        yaxis_title=NOMS["max_gap_len"],
        height=380, margin=dict(l=50, r=20, t=10, b=50),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


# ── Page principale ─────────────────────────────────────────────

def render():
    st.title("Classification RP / RS")
    st.markdown(
        "Identification des résidences principales et secondaires "
        "à partir des courbes de charge Enedis (RES2, 6-9 kVA)."
    )

    # ── Sidebar ──
    st.sidebar.markdown("### Paramètres")
    data_path = st.sidebar.text_input(
        "Chemin du CSV", value="datas/courbes-de-charges-fictives-res2-6-9.csv"
    )
    ref_path = st.sidebar.text_input(
        "Labels de référence", value="datas/RES2-6-9-labels.csv"
    )
    n_clusters = st.sidebar.slider("Nombre de clusters (k-means)", 2, 15, 5)

    # ── Chargement ──
    try:
        df, features = charger_et_traiter(data_path)
    except Exception as e:
        st.error(f"Impossible de charger les données : {e}")
        st.info("Vérifiez le chemin du fichier CSV dans la sidebar.")
        return

    # ── Calculs ──
    labels_km, clusters, scaler, km, sil = lancer_clustering(
        features, FEATURE_COLS, n_clusters
    )
    features["label"] = labels_km

    X = features[FEATURE_COLS].values
    y = labels_km
    resultats, resultats_eq = lancer_classification(X, y, FEATURE_COLS)

    # Référence (optionnel)
    ref_disponible = False
    try:
        ref = pd.read_csv(ref_path)
        ref.columns = ref.columns.str.strip()
        merged = features.merge(ref, left_on="pdl_id", right_on="id", how="inner")
        if len(merged) > 0:
            ref_disponible = True
            col_ref = "label_y" if "label_y" in merged.columns else "label"
            y_ref = merged[col_ref].values
            y_km_ref = labels_km[features["pdl_id"].isin(merged["pdl_id"])]
    except FileNotFoundError:
        pass

    # ── Métriques générales ──
    n_pdl = features["pdl_id"].nunique()
    n_rp, n_rs = (labels_km == 0).sum(), (labels_km == 1).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDL analysés", f"{n_pdl}")
    c2.metric("Mesures", f"{len(df):,}")
    c3.metric("Résidences principales", f"{n_rp} ({n_rp/n_pdl:.0%})")
    c4.metric("Résidences secondaires", f"{n_rs} ({n_rs/n_pdl:.0%})")

    # ═══════════════════════════════════════════════════════
    # ONGLETS
    # ═══════════════════════════════════════════════════════
    tab1, tab2, tab3 = st.tabs([
        "Données & Features",
        "Clustering",
        "Classification supervisée",
    ])

    # ───────────────────────────────────────────────────
    # TAB 1 : Données & Features
    # ───────────────────────────────────────────────────
    with tab1:
        st.markdown("### Exemples de courbes de charge")
        st.markdown(
            "Avant toute modélisation, on observe les données brutes. "
            "Les courbes ci-dessous montrent des profils typiques de "
            "résidence principale (consommation régulière, talon nocturne "
            "permanent) et de résidence secondaire (intermittence, longues "
            "périodes à consommation quasi nulle)."
        )

        exemples = exemples_courbes(df, features, labels_km, n_exemples=3)
        col_rp, col_rs = st.columns(2)
        with col_rp:
            st.markdown(
                f"**Résidence principale** : "
                f"occupation régulière, talon de base visible"
            )
            st.plotly_chart(
                fig_courbes_exemple(exemples, "RP", C["rp"]),
                use_container_width=True,
            )
        with col_rs:
            st.markdown(
                f"**Résidence secondaire** : "
                f"absences prolongées, pics ponctuels"
            )
            st.plotly_chart(
                fig_courbes_exemple(exemples, "RS", C["rs"]),
                use_container_width=True,
            )

        # ── Features ──
        st.markdown("---")
        st.markdown("### Construction des features")
        st.markdown(
            "À partir des courbes brutes (8,7 M de mesures), on calcule "
            "**6 indicateurs** par PDL. Chaque feature capture un aspect "
            "différent du comportement de consommation. "
            "Le choix des features repose sur l'intuition métier : "
            "une résidence secondaire se distingue par ses absences, "
            "son manque de talon nocturne, et sa saisonnalité marquée."
        )

        # Afficher chaque feature avec son histogramme et sa description
        for i in range(0, len(FEATURE_COLS), 2):
            cols = st.columns(2)
            for j, col_st in enumerate(cols):
                idx = i + j
                if idx >= len(FEATURE_COLS):
                    break
                feat = FEATURE_COLS[idx]
                with col_st:
                    st.markdown(f"**{NOMS[feat]}** (`{feat}`)")
                    st.caption(DESCRIPTIONS_FEATURES[feat])
                    st.plotly_chart(
                        fig_feature_detail(features, labels_km, feat),
                        use_container_width=True,
                    )

    # ───────────────────────────────────────────────────
    # TAB 2 : Clustering
    # ───────────────────────────────────────────────────
    with tab2:
        st.markdown("### Choix du nombre de clusters")
        st.markdown(
            "On utilise deux critères pour choisir *k* : la méthode du "
            "coude (décroissance de l'inertie) et le score silhouette "
            "(cohésion intra-cluster vs séparation inter-cluster). "
            "Le meilleur compromis se situe autour de **k = 4-6** : "
            "l'inertie ralentit sa descente et la silhouette reste élevée."
        )

        df_k = analyser_k(features, FEATURE_COLS)
        st.plotly_chart(fig_elbow_silhouette(df_k), use_container_width=True)

        st.markdown("---")
        st.markdown("### Résultat du clustering")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("k choisi", n_clusters)
            st.metric("Score silhouette", f"{sil:.3f}")

            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            cluster_occ = (
                features.groupby(clusters)["active_day_rate"].mean()
            )
            cluster_label = pd.Series(labels_km).groupby(clusters).first()

            recap = pd.DataFrame({
                "Cluster": cluster_counts.index,
                "Effectif": cluster_counts.values,
                "Taux occ. moyen": [f"{cluster_occ[c]:.2f}" for c in cluster_counts.index],
                "→ Label": ["RS" if cluster_label[c] == 1 else "RP" for c in cluster_counts.index],
            })
            st.dataframe(recap, use_container_width=True, hide_index=True)

            st.caption(
                "Le mapping cluster → label se fait automatiquement : "
                "les clusters dont le taux d'occupation moyen est inférieur "
                "à la médiane sont étiquetés RS."
            )

        with c2:
            st.plotly_chart(
                fig_scatter_clusters(features, labels_km),
                use_container_width=True,
            )

        # Comparaison avec la référence
        if ref_disponible:
            st.markdown("---")
            st.markdown("### Comparaison avec les labels de référence")
            st.markdown(
                "On compare notre clustering aux labels issus du corrigé "
                "pour valider la pertinence de notre approche."
            )
            rapport_ref, mat_ref = comparer_avec_reference(y_km_ref, y_ref)

            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.metric("Accuracy", f"{rapport_ref['accuracy']:.1%}")
                st.metric("F1 RS", f"{rapport_ref['RS']['f1-score']:.1%}")
                st.metric("Recall RS", f"{rapport_ref['RS']['recall']:.1%}")
            with c2:
                st.plotly_chart(
                    fig_matrice_confusion(mat_ref, "Clustering vs référence"),
                    use_container_width=True,
                )

    # ───────────────────────────────────────────────────
    # TAB 3 : Classification supervisée
    # ───────────────────────────────────────────────────
    with tab3:
        st.markdown("### Pourquoi un classifieur supervisé ?")
        st.markdown(
            "Le clustering labellise les PDL existants, mais ne peut pas "
            "classer un **nouveau** client sans relancer le k-means sur "
            "tout le dataset. Un classifieur supervisé, entraîné sur les "
            "labels du clustering, apprend la frontière de décision et "
            "peut prédire instantanément le type d'un nouveau PDL."
        )

        st.markdown(
            "On compare trois modèles de complexité croissante : "
            "une **régression logistique** (baseline linéaire), "
            "un **random forest** (ensemble d'arbres de décision), "
            "et un **MLP** (réseau de neurones à 2 couches cachées). "
            "Tous utilisent `class_weight='balanced'` pour compenser "
            "le déséquilibre RP/RS."
        )

        st.markdown("#### Comparaison des modèles (validation croisée 5 plis)")
        st.plotly_chart(
            fig_comparaison_modeles(resultats), use_container_width=True,
        )

        # Matrices de confusion
        st.markdown("#### Matrices de confusion")
        cols = st.columns(len(resultats))
        for col_st, (nom, res) in zip(cols, resultats.items()):
            with col_st:
                st.plotly_chart(
                    fig_matrice_confusion(res["matrice"], nom),
                    use_container_width=True,
                )

        # Importances Random Forest
        rf_res = resultats.get("Random Forest")
        if rf_res and "feature_importances" in rf_res:
            st.markdown("---")
            st.markdown("#### Importance des variables (Random Forest)")

            c1, c2 = st.columns([3, 2])
            with c1:
                st.plotly_chart(
                    fig_importances(rf_res["feature_importances"]),
                    use_container_width=True,
                )
            with c2:
                st.markdown(
                    "Le random forest permet de quantifier la contribution "
                    "de chaque feature à la décision. Les deux variables "
                    "dominantes - **durée de la plus longue absence** et "
                    "**taux d'occupation** - confirment l'intuition métier : "
                    "le premier signal d'une résidence secondaire, c'est "
                    "l'absence prolongée de consommation."
                )
                st.markdown(
                    "Le **talon nocturne** arrive en 3ᵉ position : un "
                    "logement occupé maintient un talon permanent "
                    "(réfrigérateur, veille, chauffage), contrairement "
                    "à une RS vide."
                )

        # Évaluation équilibrée
        st.markdown("---")
        st.markdown("#### Évaluation sur dataset équilibré")
        st.markdown(
            "Le dataset est déséquilibré (~85 % de RP). Pour une évaluation "
            "plus juste, on sous-échantillonne la classe majoritaire afin "
            "d'avoir autant de RP que de RS dans le jeu de test."
        )
        st.plotly_chart(
            fig_comparaison_modeles(resultats_eq), use_container_width=True,
        )

        # Synthèse
        st.markdown("---")
        st.markdown("#### Synthèse")

        meilleur = max(resultats.items(), key=lambda x: x[1]["f1_rs"])
        st.success(
            f"**Modèle retenu : {meilleur[0]}** : "
            f"Accuracy {meilleur[1]['accuracy']:.1%}, "
            f"F1 RS {meilleur[1]['f1_rs']:.1%}. "
            f"Il offre le meilleur compromis performance / interprétabilité."
        )