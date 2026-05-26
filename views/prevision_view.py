"""
Vue Prévision de consommation (CNN-LSTM) : v2
===============================================
Présentation du pipeline amélioré : normalisation par PDL, features
temporelles, split temporel, early stopping.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.prevision import pipeline_complet


# ── Palette ────────────────────────────────────────────────────

C = {
    "reel": "#2563eb",
    "pred": "#dc2626",
    "histo": "#6b7280",
    "accent": "#0ea5e9",
    "vert": "#16a34a",
    "orange": "#ea580c",
    "train": "#2563eb",
    "val": "#f59e0b",
}

JOURS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]


# ── Cache ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Entraînement du modèle en cours… (peut prendre 1-2 min)")
def lancer_pipeline(data_path, horizon, epochs, batch_size, lr):
    return pipeline_complet(
        csv_path=data_path,
        horizon=horizon,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )


# ── Graphiques ─────────────────────────────────────────────────

def fig_courbes_entrainement(historique):
    """Courbes de perte train / validation au fil des epochs."""
    fig = go.Figure()
    epochs = list(range(1, len(historique["train_loss"]) + 1))
    fig.add_trace(go.Scatter(
        x=epochs, y=historique["train_loss"],
        name="Train", line=dict(color=C["train"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=historique["val_loss"],
        name="Validation", line=dict(color=C["val"], width=2),
    ))
    # Marquer l'epoch d'arrêt
    stopped = historique.get("stopped_epoch", len(epochs))
    if stopped < len(epochs):
        fig.add_vline(
            x=stopped, line_dash="dot", line_color="#9ca3af",
            annotation_text=f"Early stop (epoch {stopped})",
        )
    fig.update_layout(
        xaxis_title="Epoch", yaxis_title="MSE (normalisée)",
        height=320, margin=dict(l=50, r=20, t=10, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def fig_prediction_individuelle(X_hist, prediction, cible):
    """Historique (H semaines) + prédiction vs réalité."""
    hist_flat = X_hist.flatten()
    n_hist = len(hist_flat)
    x_hist = list(range(n_hist))
    x_pred = list(range(n_hist, n_hist + 7))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_hist, y=hist_flat,
        name="Historique", mode="lines",
        line=dict(color=C["histo"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=x_pred, y=cible,
        name="Réalité", mode="lines+markers",
        line=dict(color=C["reel"], width=2.5),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=x_pred, y=prediction,
        name="Prédiction CNN-LSTM", mode="lines+markers",
        line=dict(color=C["pred"], width=2.5, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ))
    fig.add_vline(
        x=n_hist - 0.5, line_dash="dot",
        line_color="#9ca3af", annotation_text="Prédiction →",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Jour", yaxis_title="Consommation (kWh)",
        height=350, margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def fig_erreur_par_jour(metriques_jours):
    """Barres groupées MAE / RMSE par jour de la semaine."""
    jours = list(metriques_jours.keys())
    mae = [metriques_jours[j]["MAE"] for j in jours]
    rmse = [metriques_jours[j]["RMSE"] for j in jours]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MAE", x=jours, y=mae,
        marker_color=C["accent"],
        text=[f"{v:.2f}" for v in mae], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="RMSE", x=jours, y=rmse,
        marker_color=C["orange"],
        text=[f"{v:.2f}" for v in rmse], textposition="outside",
    ))
    fig.update_layout(
        barmode="group", yaxis_title="Erreur (kWh)",
        height=320, margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def fig_scatter_pred_reel(predictions, cibles):
    """Nuage de points prédiction vs réalité."""
    pred_flat = predictions.flatten()
    cible_flat = cibles.flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cible_flat, y=pred_flat,
        mode="markers",
        marker=dict(color=C["accent"], size=3, opacity=0.3),
        name="Échantillons",
    ))
    vmin = min(pred_flat.min(), cible_flat.min())
    vmax = max(pred_flat.max(), cible_flat.max())
    fig.add_trace(go.Scatter(
        x=[vmin, vmax], y=[vmin, vmax],
        mode="lines", name="Prédiction parfaite",
        line=dict(color=C["pred"], dash="dash", width=2),
    ))
    fig.update_layout(
        xaxis_title="Consommation réelle (kWh)",
        yaxis_title="Consommation prédite (kWh)",
        height=380, margin=dict(l=50, r=20, t=10, b=50),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def fig_distribution_erreurs(predictions, cibles):
    """Histogramme des erreurs de prédiction."""
    erreurs = (predictions - cibles).flatten()
    fig = go.Figure(go.Histogram(
        x=erreurs, nbinsx=50,
        marker_color=C["accent"], opacity=0.75,
    ))
    fig.add_vline(x=0, line_color=C["pred"], line_dash="dash", line_width=2)
    fig.update_layout(
        xaxis_title="Erreur (kWh) : prédiction − réalité",
        yaxis_title="Fréquence",
        height=300, margin=dict(l=50, r=20, t=10, b=40),
    )
    return fig


# ── Page principale ────────────────────────────────────────────

def render():
    st.title("Prévision de consommation")
    st.markdown(
        "Prédiction de la consommation électrique hebdomadaire "
        "par un réseau **CNN-LSTM**, entraîné sur les courbes de charge "
        "Enedis (RES2, 6-9 kVA)."
    )

    # ── Sidebar ──
    st.sidebar.markdown("### Paramètres du modèle")
    data_path = st.sidebar.text_input(
        "Chemin du CSV", value="datas/courbes-de-charges-fictives-res2-6-9.csv"
    )
    horizon = st.sidebar.slider("Historique (semaines)", 2, 12, 8)
    epochs = st.sidebar.slider("Epochs max", 20, 150, 80, step=10)
    lr = st.sidebar.select_slider(
        "Learning rate",
        options=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
        value=5e-4, format_func=lambda x: f"{x:.0e}",
    )

    # ── Lancement ──
    try:
        res = lancer_pipeline(data_path, horizon, epochs, 32, lr)
    except FileNotFoundError:
        st.error(f"Fichier introuvable : `{data_path}`")
        st.info(
            "Vérifiez le chemin dans la sidebar. Le CSV est attendu à la "
            "racine du projet, par exemple `datas/courbes-de-charges-fictives-res2-6-9.csv`."
        )
        return
    except KeyError as e:
        st.error(f"Colonne manquante dans le CSV : {e}")
        st.info(
            "Le fichier doit contenir au moins 3 colonnes : un identifiant "
            "client, un horodatage, et une valeur de puissance. "
            "Les séparateurs `,` et `;` sont détectés automatiquement."
        )
        try:
            import csv as _csv
            with open(data_path, "r") as f:
                head = [next(f) for _ in range(3)]
            st.code("".join(head), language="text")
        except Exception:
            pass
        return
    except Exception as e:
        st.error(f"Erreur lors du pipeline : {e}")
        st.info("Vérifiez le chemin et le format du fichier CSV dans la sidebar.")
        return

    # ── Métriques générales ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDL utilisés", f"{res['n_pdl']}")
    c2.metric("MAE", f"{res['metriques']['MAE']:.2f} kWh")
    c3.metric("RMSE", f"{res['metriques']['RMSE']:.2f} kWh")
    c4.metric("MAPE", f"{res['metriques']['MAPE']:.1f} %")

    # ═══════════════════════════════════════════════════════
    # ONGLETS
    # ═══════════════════════════════════════════════════════
    tab1, tab2, tab3 = st.tabs([
        "Données & Approche",
        "Architecture du modèle",
        "Résultats",
    ])

    # ───────────────────────────────────────────────────
    # TAB 1 : Données & Approche
    # ───────────────────────────────────────────────────
    with tab1:
        st.markdown("### Du signal brut aux séquences d'entraînement")
        st.markdown(
            "Les données brutes sont des mesures de puissance (W) au pas de "
            "30 minutes. Le pipeline les transforme en trois étapes avant "
            "de les passer au modèle."
        )

        st.markdown("#### 1. Agrégation journalière")
        st.markdown(
            "Chaque mesure de puissance est multipliée par 0,5 h (le pas "
            "de temps), puis sommée sur la journée et convertie en **kWh**. "
            "On passe de ~17 500 points/PDL/an à ~365 valeurs journalières."
        )

        st.markdown("#### 2. Normalisation par PDL")
        st.markdown(
            "Au lieu d'un z-score "
            "global (tous les PDL mélangés), **chaque PDL est normalisé "
            "individuellement** par sa propre moyenne et son écart-type."
        )
        st.markdown(
            "**Pourquoi ?** Un studio à 4 kWh/jour et un logement chauffage "
            "électrique à 20 kWh/jour ont des échelles très différentes. "
            "Avec un z-score global, le modèle devait apprendre à la fois "
            "le *niveau* de consommation et la *forme* de la courbe. "
            "En normalisant par PDL, il se concentre uniquement sur la forme : "
            "« cette semaine est 30 % au-dessus de la moyenne du client ». "
            "La dénormalisation se fait après prédiction, en multipliant par "
            "le σ du PDL et en ajoutant sa moyenne."
        )

        st.markdown("#### 3. Fenêtrage glissant + features temporelles")
        st.markdown(
            f"La série de chaque PDL est découpée en semaines (7 jours). "
            f"Une fenêtre glissante prend **{horizon} semaines** en entrée et "
            f"prédit la **semaine suivante**."
        )

        col_schema, col_stats = st.columns([3, 2])
        with col_schema:
            st.markdown(
                "```\n"
                f"  Entrée ({horizon} sem × 9 features)      Sortie\n"
                f" ┌────────────────────────────────┐  ┌──────────┐\n"
                f" │ 7 conso + sin(t) + cos(t)      │→ │ 7 jours  │\n"
                f" │ par semaine, × {horizon} semaines      │  │ prédits  │\n"
                f" └────────────────────────────────┘  └──────────┘\n"
                "```"
            )
            st.caption(
                "sin(t) et cos(t) encodent la position dans l'année "
                "(saisonnalité). Ainsi le modèle sait « on est en janvier » "
                "même s'il ne voit que 8 semaines d'historique."
            )

        with col_stats:
            st.metric("Échantillons train", f"{res['n_samples_train']:,}")
            st.metric("Échantillons test", f"{res['n_samples_test']:,}")
            st.metric("Échantillons validation", f"{res['n_samples_val']:,}")

        st.markdown("---")
        st.markdown("#### Pourquoi 8 semaines et pas 4 ? Et pourquoi pas 52 ?")
        st.markdown(
            f"Avec seulement 4 semaines d'historique, le modèle ne voyait "
            f"qu'un mois de recul : insuffisant pour capter les tendances "
            f"(montée progressive du chauffage en automne, baisse au printemps). "
            f"**{horizon} semaines** ({horizon * 7} jours ≈ {horizon // 4} mois) "
            f"offre un meilleur compromis."
        )
        st.markdown(
            "Pourquoi ne pas utiliser l'année entière (52 semaines) ? "
            "Parce qu'avec un dataset d'un an, il ne resterait presque plus "
            "rien à prédire. De plus, une matrice 52×7 serait très grande pour "
            "le CNN et allongerait fortement l'entraînement. Les **features "
            "temporelles** (sin/cos) résolvent ce problème autrement : elles "
            "injectent l'information saisonnière *directement* dans l'input, "
            "sans avoir besoin de « voir » toute l'année."
        )

        st.markdown("---")
        st.markdown("#### Split temporel (pas random)")
        st.markdown(
            "Le split est **chronologique** par PDL : les premières "
            "fenêtres vont en train, les suivantes en validation, les dernières "
            "en test. Cela simule un usage réel (on prédit l'avenir, pas le passé) "
            "et donne des métriques plus honnêtes."
        )

    # ───────────────────────────────────────────────────
    # TAB 2 : Architecture du modèle
    # ───────────────────────────────────────────────────
    with tab2:
        st.markdown("### CNN-LSTM : extraction de motifs + mémoire temporelle")
        st.markdown(
            "L'architecture hybride CNN-LSTM tire parti de deux mécanismes "
            "complémentaires."
        )

        col_desc, col_detail = st.columns([3, 2])
        with col_desc:
            st.markdown("#### Partie CNN (Convolutionnel)")
            st.markdown(
                "Deux couches de convolution 2D (16 puis 32 filtres, noyau 3×3) "
                "parcourent la matrice d'entrée pour extraire des **motifs "
                "locaux** : pics de consommation le matin, creux la nuit, "
                "différences semaine/week-end. Chaque couche est suivie de "
                "BatchNorm (stabilise l'entraînement) et ReLU (non-linéarité)."
            )

            st.markdown("#### Partie LSTM (Récurrent)")
            st.markdown(
                "Les features extraites par le CNN sont réorganisées en séquence "
                "temporelle (une étape par semaine) et passées à un **LSTM "
                "à 2 couches** (128 neurones cachés). Le LSTM apprend les "
                "**dépendances inter-semaines** : tendances, montée du chauffage, "
                "effet vacances. Les 2 couches permettent de capter des "
                "dépendances plus abstraites (1 couche, 64 neurones)."
            )

            st.markdown("#### Tête de prédiction")
            st.markdown(
                "Le dernier état caché du LSTM passe par deux couches denses "
                "(128 -> 64 -> 7) avec dropout entre les deux. La sortie est "
                "un vecteur de 7 valeurs : la consommation prédite (normalisée) "
                "pour chaque jour de la semaine cible."
            )

        with col_detail:
            st.markdown("#### Récapitulatif")
            archi = pd.DataFrame([
                {"Couche": "Conv2D (1→16)", "Sortie": f"(16, {horizon}, 9)", "Rôle": "Motifs locaux"},
                {"Couche": "BatchNorm + ReLU", "Sortie": ":", "Rôle": "Stabilisation"},
                {"Couche": "Conv2D (16→32)", "Sortie": f"(32, {horizon}, 9)", "Rôle": "Motifs abstraits"},
                {"Couche": "BatchNorm + ReLU", "Sortie": ":", "Rôle": "Stabilisation"},
                {"Couche": "Dropout (0.15)", "Sortie": ":", "Rôle": "Régularisation"},
                {"Couche": "LSTM 2 couches", "Sortie": f"({horizon}, 128)", "Rôle": "Séquence temporelle"},
                {"Couche": "Dense (128→64)", "Sortie": "(64,)", "Rôle": "Projection"},
                {"Couche": "Dropout (0.2)", "Sortie": ":", "Rôle": "Régularisation"},
                {"Couche": "Dense (64→7)", "Sortie": "(7,)", "Rôle": "Prédiction"},
            ])
            st.dataframe(archi, use_container_width=True, hide_index=True)

            total_params = sum(p.numel() for p in res["model"].parameters())
            st.metric("Paramètres total", f"{total_params:,}")
            st.metric("Epochs (early stop)", res["epochs"])

        # Courbe d'entraînement
        st.markdown("---")
        st.markdown("#### Courbe d'entraînement")
        st.markdown(
            "L'évolution de la perte (MSE) pendant l'entraînement. "
            "L'**early stopping** arrête automatiquement si la validation "
            "ne s'améliore plus pendant 12 epochs, et restaure le meilleur "
            "modèle observé. Cela évite le sur-apprentissage."
        )
        st.plotly_chart(
            fig_courbes_entrainement(res["historique"]),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("#### Pourquoi CNN-LSTM ?")
        st.markdown(
            "Un **CNN seul** capture les motifs intra-semaine (patterns "
            "jour/nuit, semaine/week-end) mais ignore l'ordre des semaines. "
            "Un **LSTM seul** voit la séquence temporelle mais peine avec "
            "les structures fines. Le **CNN-LSTM** combine les deux : "
            "le CNN encode chaque semaine en vecteur de features, le LSTM "
            "les lit dans l'ordre chronologique. C'est une architecture "
            "standard et bien documentée en prévision de séries temporelles."
        )

    # ───────────────────────────────────────────────────
    # TAB 3 : Résultats
    # ───────────────────────────────────────────────────
    with tab3:
        st.markdown("### Performance globale")

        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("#### Prédiction vs Réalité")
            st.markdown(
                "Chaque point = une prédiction journalière. Plus les points "
                "sont proches de la diagonale, meilleur est le modèle."
            )
            st.plotly_chart(
                fig_scatter_pred_reel(res["predictions"], res["cibles"]),
                use_container_width=True,
            )
        with c2:
            st.markdown("#### Distribution des erreurs")
            st.markdown(
                "L'histogramme doit être centré sur 0 (pas de biais) "
                "et le plus resserré possible."
            )
            st.plotly_chart(
                fig_distribution_erreurs(res["predictions"], res["cibles"]),
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("#### Erreur par jour de la semaine")
        st.markdown(
            "Le modèle prédit-il aussi bien le lundi que le dimanche ? "
            "Les week-ends sont souvent plus difficiles (comportements "
            "moins réguliers)."
        )
        st.plotly_chart(
            fig_erreur_par_jour(res["metriques_jours"]),
            use_container_width=True,
        )

        # ── Exemples individuels ──
        st.markdown("---")
        st.markdown("### Exploration par échantillon")
        st.markdown(
            "Sélectionnez un échantillon de test pour visualiser "
            "l'historique, la courbe réelle et la prédiction."
        )

        n_test = len(res["predictions"])
        idx = st.slider("Échantillon de test", 0, n_test - 1, 0)

        pdl_id, sem_cible = res["meta_test"][idx]
        st.caption(f"PDL `{pdl_id}` : semaine cible n°{sem_cible}")

        st.plotly_chart(
            fig_prediction_individuelle(
                res["X_test"][idx],
                res["predictions"][idx],
                res["cibles"][idx],
            ),
            use_container_width=True,
        )

        detail = pd.DataFrame({
            "Jour": JOURS,
            "Réel (kWh)": [f"{v:.2f}" for v in res["cibles"][idx]],
            "Prédit (kWh)": [f"{v:.2f}" for v in res["predictions"][idx]],
            "Erreur (kWh)": [
                f"{res['predictions'][idx][j] - res['cibles'][idx][j]:+.2f}"
                for j in range(7)
            ],
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)

        # ── Synthèse ──
        st.markdown("---")
        st.markdown("### Synthèse")

        metr = res["metriques"]
        # Adapter le message au niveau de performance
        if metr["MAPE"] < 15:
            qualite = "bon"
            icone = "success"
        elif metr["MAPE"] < 25:
            qualite = "correct"
            icone = "success"
        else:
            qualite = "perfectible"
            icone = "warning"

        msg = (
            f"**CNN-LSTM ({horizon} sem. → 1 sem.)** : "
            f"MAE = {metr['MAE']:.2f} kWh, "
            f"RMSE = {metr['RMSE']:.2f} kWh, "
            f"MAPE = {metr['MAPE']:.1f} %. "
        )

        if icone == "success":
            st.success(
                msg + "Le modèle offre une précision "
                f"{qualite} pour de la prévision résidentielle."
            )
        else:
            st.warning(
                msg + "La performance est perfectible, ce qui est attendu "
                "en résidentiel individuel (forte variabilité des comportements). "
                "Les pistes d'amélioration incluent un historique plus long, "
                "des features météo, ou un modèle par cluster de PDL."
            )

        st.caption(
            "**MAE** = erreur moyenne absolue, interprétable en kWh. "
            "**RMSE** = pénalise les grosses erreurs. "
            "**MAPE** = erreur relative (%), calculée en excluant les jours "
            "< 1 kWh (RS vides qui gonflent artificiellement l'erreur). "
            "En résidentiel individuel, un MAPE < 15 % est bon, < 25 % correct."
        )