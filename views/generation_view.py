"""
Vue Génération de courbes de charge — VAE conditionnel
================================================================
Trois onglets : Approche, Modèle, Résultats.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.generation import pipeline_complet, generer


C = {"rp": "#2563eb", "rs": "#dc2626", "gen_rp": "#60a5fa",
     "gen_rs": "#f87171", "train": "#2563eb", "val": "#f59e0b"}
HEURES = [f"{h}:{m:02d}" for h in range(24) for m in (0, 30)]


@st.cache_data(show_spinner="Entraînement du VAE…")
def lancer_pipeline(csv_path, labels_path, epochs, batch_size, lr, beta):
    return pipeline_complet(csv_path, labels_path, epochs=epochs,
                            batch_size=batch_size, lr=lr, beta=beta)


# ── Graphiques ─────────────────────────────────────────────────

def fig_entrainement(hist):
    fig = go.Figure()
    ep = list(range(1, len(hist["train"]) + 1))
    fig.add_trace(go.Scatter(x=ep, y=hist["train"], name="Train",
        line=dict(color=C["train"], width=2)))
    fig.add_trace(go.Scatter(x=ep, y=hist["val"], name="Validation",
        line=dict(color=C["val"], width=2)))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Perte totale",
        height=300, margin=dict(l=50, r=20, t=10, b=40),
        legend=dict(orientation="h", y=1.1))
    return fig


def fig_courbes(reels, generes, label_type, couleur_reel, couleur_gen):
    """Superpose quelques courbes réelles et générées pour comparer
    visuellement la forme des profils."""
    fig = go.Figure()
    n_show = min(8, len(reels), len(generes))

    for i in range(n_show):
        fig.add_trace(go.Scatter(
            x=HEURES, y=reels[i], mode="lines",
            line=dict(color=couleur_reel, width=1),
            opacity=0.4, name="Réel" if i == 0 else None,
            showlegend=(i == 0), legendgroup="reel"))

    for i in range(n_show):
        fig.add_trace(go.Scatter(
            x=HEURES, y=generes[i], mode="lines",
            line=dict(color=couleur_gen, width=1.5, dash="dash"),
            opacity=0.6, name="Généré" if i == 0 else None,
            showlegend=(i == 0), legendgroup="gen"))

    fig.update_layout(
        xaxis_title="Heure de la journée", yaxis_title="Puissance (W)",
        height=350, margin=dict(l=50, r=20, t=30, b=40),
        title=dict(text=f"Profils {label_type}", x=0.5, font=dict(size=14)),
        legend=dict(orientation="h", y=1.1))
    return fig


# ── Page ───────────────────────────────────────────────────────

def render():
    st.title("Génération de courbes de charge")
    st.markdown(
        "Génération de profils journaliers synthétiques à l'aide d'un "
        "**auto-encodeur variationnel conditionnel** (CVAE), capable de "
        "produire des courbes RP ou RS à la demande."
    )

    st.sidebar.markdown("### Paramètres")
    csv_path = st.sidebar.text_input(
        "Chemin du CSV", value="datas/courbes-de-charges-fictives-res2-6-9.csv")
    labels_path = st.sidebar.text_input(
        "Fichier de labels", value="datas/RES2-6-9-labels.csv")
    epochs = st.sidebar.slider("Epochs max", 30, 300, 120, step=10)
    beta = st.sidebar.select_slider("β (poids KL)",
        options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)

    try:
        res = lancer_pipeline(csv_path, labels_path, epochs, 64, 1e-3, beta)
    except FileNotFoundError as e:
        st.error(f"Fichier introuvable : {e}")
        return
    except Exception as e:
        st.error(f"Erreur : {e}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Profils journaliers", f"{res['n_profils']:,}")
    c2.metric("RP", f"{res['n_rp']:,}")
    c3.metric("RS", f"{res['n_rs']:,}")

    tab1, tab2, tab3 = st.tabs(["Approche", "Modèle", "Résultats"])

    # ───────────────── TAB 1 ─────────────────
    with tab1:
        st.markdown("### Pourquoi générer des courbes ?")
        st.markdown(
            "Les données de consommation réelles sont soumises à des "
            "contraintes de confidentialité et ne sont pas toujours "
            "disponibles en quantité suffisante. Pouvoir **générer des "
            "courbes synthétiques réalistes** permet de tester des algorithmes, "
            "d'enrichir un jeu d'entraînement ou de partager des données "
            "sans compromettre la vie privée des consommateurs."
        )

        st.markdown("### Le principe du VAE")
        st.markdown(
            "Un auto-encodeur classique compresse des données puis les "
            "reconstruit. Le problème : son espace latent (la représentation "
            "compressée) est désorganisé, on ne peut pas y tirer des points "
            "au hasard pour générer quelque chose de cohérent."
        )
        st.markdown(
            "Le VAE ajoute une contrainte : l'espace latent doit "
            "ressembler à une gaussienne. Cette régularisation fait que "
            "des points tirés aléatoirement dans cet espace correspondent "
            "à des courbes plausibles."
        )

        st.markdown("### Conditionnement RP / RS")
        st.markdown(
            "On veut pouvoir choisir le type de courbe à générer. Pour ça, "
            "on concatène le label (0 = RP, 1 = RS) à l'entrée de l'encodeur "
            "et du décodeur. Le modèle apprend ainsi à associer certaines "
            "zones de l'espace latent à chaque type de résidence."
        )

        st.markdown("---")
        st.markdown("### Les données")
        st.markdown(
            "On travaille sur des **profils journaliers** : 48 mesures de "
            "puissance (en watts) couvrant une journée complète au pas de "
            "30 minutes. Chaque foyer contribue autant de profils qu'il a "
            "de jours complets dans le dataset."
        )
        st.markdown(
            "Les profils sont normalisés (centrés-réduits) avant "
            "d'entrer dans le modèle, puis remis en watts pour l'affichage."
        )

    # ───────────────── TAB 2 ─────────────────
    with tab2:
        st.markdown("### Architecture du CVAE")
        col_d, col_t = st.columns([3, 2])
        with col_d:
            st.markdown(
                "L'encodeur est un réseau dense à deux couches cachées "
                "(128 puis 64 neurones). Il prend en entrée un profil de "
                "48 valeurs + le label (49 entrées au total) et produit "
                "deux vecteurs de taille 8 : la **moyenne** et la **variance** "
                "de la distribution latente."
            )
            st.markdown(
                "Le décodeur a la structure inverse : il prend un point "
                "latent de taille 8 + le label (9 entrées) et reconstruit "
                "un profil de 48 valeurs."
            )
        with col_t:
            archi = pd.DataFrame([
                {"Bloc": "Encodeur", "Couches": "49 → 128 → 64", "Sortie": "μ (8) + σ² (8)"},
                {"Bloc": "Latent", "Couches": "reparamétrisation", "Sortie": "z (8)"},
                {"Bloc": "Décodeur", "Couches": "9 → 64 → 128", "Sortie": "profil (48)"},
            ])
            st.dataframe(archi, use_container_width=True, hide_index=True)
            tot = sum(p.numel() for p in res["model"].parameters())
            st.metric("Paramètres", f"{tot:,}")

        st.markdown("#### Fonction de perte")
        st.markdown(
            "Deux termes additionnés. La **reconstruction** (MSE) mesure "
            "la fidélité de la courbe reconstruite par rapport à l'originale. "
            "La **divergence KL** force l'espace latent à rester proche d'une "
            "gaussienne standard, ce qui garantit qu'un tirage aléatoire "
            "dans cet espace produit des courbes cohérentes."
        )

        st.markdown("#### Courbe d'entraînement")
        st.plotly_chart(fig_entrainement(res["historique"]),
                        use_container_width=True)

    # ───────────────── TAB 3 ─────────────────
    with tab3:
        st.markdown("### Courbes générées vs courbes réelles")
        st.markdown(
            "On compare visuellement des profils tirés du modèle (en "
            "pointillés) avec des profils réels (en trait plein). Les "
            "courbes générées doivent avoir la même allure générale : "
            "double pic matin/soir pour les RP, profil plus plat ou "
            "irrégulier pour les RS."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                fig_courbes(res["reels_rp"], res["gen_rp"], "RP",
                            C["rp"], C["gen_rp"]),
                use_container_width=True)
        with col2:
            st.plotly_chart(
                fig_courbes(res["reels_rs"], res["gen_rs"], "RS",
                            C["rs"], C["gen_rs"]),
                use_container_width=True)

        st.markdown("### Comparaison statistique")
        st.markdown(
            "Au-delà du visuel, on vérifie que les grandeurs moyennes "
            "(puissance globale, pic matin, pic soir, creux nocturne) "
            "correspondent entre courbes réelles et générées."
        )

        def table_comp(comp, label):
            r, g = comp["reel"], comp["genere"]
            return pd.DataFrame({
                "Grandeur": ["Moyenne (W)", "Écart-type (W)",
                             "Pic matin (W)", "Pic soir (W)", "Creux nuit (W)"],
                "Réel": [f"{r['moyenne']:.0f}", f"{r['ecart_type']:.0f}",
                         f"{r['pic_matin']:.0f}", f"{r['pic_soir']:.0f}",
                         f"{r['creux_nuit']:.0f}"],
                "Généré": [f"{g['moyenne']:.0f}", f"{g['ecart_type']:.0f}",
                           f"{g['pic_matin']:.0f}", f"{g['pic_soir']:.0f}",
                           f"{g['creux_nuit']:.0f}"],
            })

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Résidences principales**")
            st.dataframe(table_comp(res["comp_rp"], "RP"),
                         use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Résidences secondaires**")
            st.dataframe(table_comp(res["comp_rs"], "RS"),
                         use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Génération à la demande")
        st.markdown("Tirez de nouvelles courbes en choisissant le type.")

        col_ctrl = st.columns(3)
        type_res = col_ctrl[0].selectbox("Type", ["RP", "RS"])
        n_gen = col_ctrl[1].slider("Nombre", 1, 30, 10)

        if col_ctrl[2].button("Générer"):
            label_val = 0 if type_res == "RP" else 1
            nouvelles = generer(res["model"], label_val, n_gen, res["stats"])
            fig = go.Figure()
            for i in range(n_gen):
                fig.add_trace(go.Scatter(
                    x=HEURES, y=nouvelles[i], mode="lines",
                    line=dict(width=1.5),
                    name=f"#{i+1}", showlegend=False))
            fig.update_layout(
                xaxis_title="Heure", yaxis_title="Puissance (W)",
                height=350, margin=dict(l=50, r=20, t=10, b=40))
            st.plotly_chart(fig, use_container_width=True)