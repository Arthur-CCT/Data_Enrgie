"""
Vue Prévision de consommation
================================================================
Trois onglets :
  1. Approche   : l'idée (niveau × forme) expliquée simplement
  2. Modèle     : ce que fait le CNN-LSTM
  3. Résultats  : performance et exemple concret
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.prevision import pipeline_complet


C = {
    "reel": "#2563eb", "pred": "#dc2626", "histo": "#9ca3af",
    "baseline": "#8b5cf6", "train": "#2563eb", "val": "#f59e0b",
}
JOURS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]


@st.cache_data(show_spinner="Entraînement du modèle… (quelques minutes)")
def lancer_pipeline(data_path, horizon, epochs, batch_size, lr):
    return pipeline_complet(csv_path=data_path, horizon=horizon,
                            epochs=epochs, batch_size=batch_size, lr=lr)


# ── Les deux seuls graphiques de la page ───────────────────────

def fig_entrainement(hist):
    """Courbe d'apprentissage : l'erreur doit baisser sur train ET validation."""
    fig = go.Figure()
    ep = list(range(1, len(hist["train_loss"]) + 1))
    fig.add_trace(go.Scatter(x=ep, y=hist["train_loss"], name="Train",
        line=dict(color=C["train"], width=2)))
    fig.add_trace(go.Scatter(x=ep, y=hist["val_loss"], name="Validation",
        line=dict(color=C["val"], width=2)))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Erreur",
        height=300, margin=dict(l=50, r=20, t=10, b=40),
        legend=dict(orientation="h", y=1.1))
    return fig


def fig_exemple(hist_raw, pred, reel):
    """Un cas concret : historique, semaine réelle et semaine prédite."""
    hist_flat = hist_raw.flatten()
    n = len(hist_flat)
    xh, xp = list(range(n)), list(range(n, n + 7))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xh, y=hist_flat, name="Historique",
        mode="lines", line=dict(color=C["histo"], width=1.5)))
    fig.add_trace(go.Scatter(x=xp, y=reel, name="Réalité",
        mode="lines+markers", line=dict(color=C["reel"], width=2.5),
        marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=xp, y=pred, name="Prédiction",
        mode="lines+markers", line=dict(color=C["pred"], width=2.5, dash="dash"),
        marker=dict(size=6, symbol="diamond")))
    fig.add_vline(x=n - 0.5, line_dash="dot", line_color="#9ca3af",
        annotation_text="Prédiction →", annotation_position="top right")
    fig.update_layout(xaxis_title="Jour", yaxis_title="Consommation (kWh)",
        height=360, margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", y=1.12))
    return fig


def table_comparaison(metr_p, metr_b, metr_m):
    """Petit tableau : modèle vs méthodes naïves."""
    return pd.DataFrame({
        "Méthode": [metr_p["nom"], metr_b["nom"], metr_m["nom"]],
        "MAPE": [f"{metr_p['MAPE']:.1f} %", f"{metr_b['MAPE']:.1f} %", f"{metr_m['MAPE']:.1f} %"],
        "MAE (kWh)": [f"{metr_p['MAE']:.2f}", f"{metr_b['MAE']:.2f}", f"{metr_m['MAE']:.2f}"],
    })


# ── Page ───────────────────────────────────────────────────────

def render():
    st.title("Prévision de consommation")
    st.markdown(
        "On prédit la consommation de la semaine à venir d'un foyer à partir "
        "de son historique, avec un réseau **CNN-LSTM**. Données Enedis "
        "(RES2, 6-9 kVA)."
    )

    # Paramètres
    st.sidebar.markdown("### Paramètres du modèle")
    data_path = st.sidebar.text_input(
        "Chemin du CSV", value="datas/courbes-de-charges-fictives-res2-6-9.csv")
    horizon = st.sidebar.slider("Historique (semaines)", 4, 12, 8)
    epochs = st.sidebar.slider("Epochs max", 30, 200, 100, step=10)
    lr = st.sidebar.select_slider("Learning rate",
        options=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3], value=5e-4,
        format_func=lambda x: f"{x:.0e}")

    try:
        res = lancer_pipeline(data_path, horizon, epochs, 32, lr)
    except FileNotFoundError:
        st.error(f"Fichier introuvable : `{data_path}`"); return
    except KeyError as e:
        st.error(f"Colonne manquante : {e}"); return
    except Exception as e:
        st.error(f"Erreur : {e}"); return

    mt, md = res["metr_total"], res["metr_daily"]

    # Bannière : les chiffres clés
    c1, c2, c3 = st.columns(3)
    c1.metric("MAPE : total de la semaine", f"{mt['MAPE']:.1f} %")
    c2.metric("MAPE : jour par jour", f"{md['MAPE']:.1f} %")
    c3.metric("Foyers analysés", f"{res['n_pdl']}")

    tab1, tab2, tab3 = st.tabs(["Approche", "Modèle", "Résultats"])

    # ───────────────── TAB 1 : APPROCHE ─────────────────
    with tab1:
        st.markdown("### Le problème")
        st.markdown(
            "Prédire la conso **d'un jour précis** pour **un seul foyer** est "
            "très difficile : un jour dépend de ce que font les habitants "
            "(sorties, invités, télétravail…), ce qui est en grande partie "
            "imprévisible. Quel que soit le modèle, l'erreur jour par jour "
            "reste élevée."
        )

        st.markdown("### L'idée : séparer en deux")
        st.markdown(
            "Plutôt que d'attaquer les 7 jours de front, on découpe la "
            "prévision en deux quantités plus simples à prédire :"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### NIVEAU")
            st.markdown(
                "L'énergie **totale** de la semaine (un seul nombre). "
                "En additionnant les 7 jours, les aléas de chaque jour se "
                "compensent : c'est nettement plus prévisible."
            )
        with col_b:
            st.markdown("#### FORME")
            st.markdown(
                "La **répartition** de cette énergie sur les 7 jours, "
                "c'est-à-dire l'habitude du foyer (creux en semaine, pic le "
                "week-end). Très stable d'une semaine à l'autre."
            )
        st.markdown(
            "> **Prévision d'un jour = NIVEAU prédit × part du jour dans la FORME**"
        )
        st.markdown(
            "Le modèle ne s'occupe que du NIVEAU. La FORME est lue directement "
            "dans l'historique du foyer."
        )

        st.markdown("### Une astuce pour le NIVEAU : prédire l'écart")
        st.markdown(
            "On ne demande pas au modèle le total brut, mais l'**écart** par "
            "rapport à une estimation simple (la moyenne des semaines récentes). "
            "Apprendre une petite correction est plus facile que tout deviner : "
            "même si le modèle se trompe, on retombe sur une valeur raisonnable."
        )

        st.markdown("### En résumé, les ingrédients")
        st.markdown(
            "- **Énergie par jour** : on agrège la puissance (pas de 30 min) en kWh/jour.\n"
            "- **Même échelle pour tous les foyers** : on centre/réduit chaque foyer.\n"
            "- **Repère de saison** : sin/cos du jour de l'année (chauffage l'hiver).\n"
            "- **Découpage dans l'ordre du temps** : on n'entraîne jamais sur le futur du test."
        )

        st.markdown("---")
        st.markdown("### Les données et leur découpage")
        st.markdown(
            f"Courbes de charge Enedis (RES2, 6-9 kVA) : la puissance relevée "
            f"toutes les 30 minutes sur un an, pour **{res['n_pdl']} foyers** "
            f"retenus (ceux ayant assez d'historique). Une fois découpées en "
            f"fenêtres de {horizon} semaines, on obtient des exemples répartis "
            f"**dans l'ordre du temps** pour chaque foyer :"
        )
        cs = st.columns(3)
        cs[0].metric("Entraînement", f"{res['n_train']:,}",
                     help="Premières semaines : le modèle apprend dessus.")
        cs[1].metric("Validation", f"{res['n_val']:,}",
                     help="Semaines suivantes : règlent l'arrêt de l'entraînement.")
        cs[2].metric("Test", f"{res['n_test']:,}",
                     help="Dernières semaines : jamais vues, servent à mesurer la performance.")
        st.caption(
            "Train, validation et test sont pris dans cet ordre chronologique "
            "(pas au hasard) pour que le test porte toujours sur des semaines "
            "postérieures à l'entraînement, comme en situation réelle."
        )

    # ───────────────── TAB 2 : MODÈLE ─────────────────
    with tab2:
        st.markdown("### Ce que fait le CNN-LSTM")
        st.markdown(
            "Le modèle reçoit l'historique sous forme d'un tableau "
            f"(**{horizon} semaines × 9 colonnes** : 7 jours + 2 repères de "
            "saison) et renvoie **un seul nombre** : la correction de NIVEAU."
        )

        col_d, col_t = st.columns([3, 2])
        with col_d:
            st.markdown(
                "- **CNN** : deux couches de convolution repèrent les motifs "
                "à l'intérieur d'une semaine (différence semaine / week-end).\n"
                "- **LSTM** : lit les semaines dans l'ordre et capte la "
                "tendance (la conso monte-t-elle, baisse-t-elle ?).\n"
                "- **Sortie** : deux couches denses produisent la correction "
                "de niveau."
            )
        with col_t:
            archi = pd.DataFrame([
                {"Étape": "CNN (conv ×2)", "Sortie": f"(32, {horizon}, 9)"},
                {"Étape": "LSTM (×2)", "Sortie": f"({horizon}, 128)"},
                {"Étape": "Dense", "Sortie": "1 nombre"},
            ])
            st.dataframe(archi, use_container_width=True, hide_index=True)
            tot = sum(p.numel() for p in res["model"].parameters())
            st.metric("Paramètres", f"{tot:,}")

        st.markdown("#### Apprentissage")
        st.markdown(
            "L'entraînement s'arrête tout seul quand l'erreur de validation "
            "cesse de baisser (*early stopping*), pour éviter le sur-apprentissage."
        )
        st.plotly_chart(fig_entrainement(res["historique"]), use_container_width=True)

    # ───────────────── TAB 3 : RÉSULTATS ─────────────────
    with tab3:
        st.markdown("### Performance")
        st.markdown(
            "On compare le modèle à deux méthodes naïves : la **persistence** "
            "(« la semaine prochaine sera comme la dernière ») et la "
            "**baseline** (moyenne des semaines récentes). Un modèle utile "
            "doit faire mieux qu'elles."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Total de la semaine**")
            st.dataframe(
                table_comparaison(res["metr_total_persist"],
                                  res["metr_total_base"], res["metr_total"]),
                use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Jour par jour**")
            st.dataframe(
                table_comparaison(res["metr_daily_persist"],
                                  res["metr_daily_base"], res["metr_daily"]),
                use_container_width=True, hide_index=True)

        st.markdown(
            "Le total de la semaine est bien mieux prédit que le détail jour "
            "par jour : c'est attendu, c'est la quantité réellement prévisible "
            "(et la plus utile pour la gestion de réseau)."
        )

        st.markdown("---")
        st.markdown("### Un exemple")
        st.markdown(
            "Choisissez une semaine de test pour voir l'historique, la réalité "
            "et la prédiction."
        )
        n_test = len(res["daily_pred"])
        idx = st.slider("Semaine de test", 0, n_test - 1, 0)
        pid, sem = res["meta_test"][idx]
        st.caption(
            f"Foyer `{pid}` — total réel {res['total_reel'][idx]:.1f} kWh, "
            f"prédit {res['total_pred'][idx]:.1f} kWh"
        )
        st.plotly_chart(
            fig_exemple(res["X_test_raw"][idx], res["daily_pred"][idx],
                        res["daily_reel"][idx]),
            use_container_width=True)

        detail = pd.DataFrame({
            "Jour": JOURS,
            "Réel (kWh)": [f"{v:.2f}" for v in res["daily_reel"][idx]],
            "Prédit (kWh)": [f"{v:.2f}" for v in res["daily_pred"][idx]],
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)

        st.caption(
            "MAPE = erreur moyenne en %. MAE = erreur moyenne en kWh. "
            "Calculé sur les valeurs > 2 kWh (on écarte les résidences "
            "secondaires vides, qui fausseraient le %)."
        )