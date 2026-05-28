"""
Vue Génération de courbes de charge
================================================================
Trois onglets : Approche, Modèles, Résultats.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.generation import pipeline_complet, generer


C = {"rp": "#2563eb", "rs": "#dc2626", "gen": "#f59e0b",
     "lin": "#8b5cf6", "conv": "#0ea5e9"}


@st.cache_data(show_spinner="Entraînement des deux VAE…")
def lancer(csv, lab, epochs, batch, lr, beta):
    return pipeline_complet(csv, lab, epochs, batch, lr, beta)


def fig_training(hist_lin, hist_conv):
    fig = go.Figure()
    for hist, nom, col, dash in [
        (hist_lin, "Linéaire – train", C["lin"], "solid"),
        (hist_lin, "Linéaire – val", C["lin"], "dash"),
        (hist_conv, "Conv-Att – train", C["conv"], "solid"),
        (hist_conv, "Conv-Att – val", C["conv"], "dash"),
    ]:
        key = "train" if "train" in nom else "val"
        fig.add_trace(go.Scatter(
            x=list(range(1, len(hist[key]) + 1)), y=hist[key],
            name=nom, line=dict(color=col, width=2, dash=dash)))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Perte",
        height=320, margin=dict(l=50, r=20, t=10, b=40),
        legend=dict(orientation="h", y=1.15))
    return fig


def fig_courbes(reels, generes, titre, c_reel, c_gen):
    fig = go.Figure()
    n = min(5, len(reels), len(generes))
    for i in range(n):
        fig.add_trace(go.Scatter(y=reels[i], mode="lines",
            line=dict(color=c_reel, width=1), opacity=0.4,
            name="Réel" if i == 0 else None,
            showlegend=(i == 0), legendgroup="r"))
    for i in range(n):
        fig.add_trace(go.Scatter(y=generes[i], mode="lines",
            line=dict(color=c_gen, width=1.5, dash="dash"), opacity=0.7,
            name="Généré" if i == 0 else None,
            showlegend=(i == 0), legendgroup="g"))
    fig.update_layout(xaxis_title="Jour de l'année", yaxis_title="kWh/jour",
        height=320, margin=dict(l=50, r=20, t=30, b=40),
        title=dict(text=titre, x=0.5, font=dict(size=14)),
        legend=dict(orientation="h", y=1.1))
    return fig


def render():
    st.title("Génération de courbes de charge")
    st.markdown(
        "Génération de profils annuels synthétiques avec un "
        "**VAE conditionnel** (RP ou RS). On compare deux architectures : "
        "un modèle linéaire et un modèle Conv-Attention."
    )

    st.sidebar.markdown("### Paramètres")
    csv_path = st.sidebar.text_input(
        "CSV consommation", value="datas/courbes-de-charges-fictives-res2-6-9.csv")
    labels_path = st.sidebar.text_input(
        "CSV labels", value="datas/RES2-6-9-labels.csv")
    epochs = st.sidebar.slider("Epochs max", 50, 500, 200, step=50)
    beta = st.sidebar.select_slider("β (poids KL)",
        options=[0.01, 0.05, 0.1, 0.5, 1.0], value=0.05)

    try:
        res = lancer(csv_path, labels_path, epochs, 32, 1e-3, beta)
    except FileNotFoundError as e:
        st.error(f"Fichier introuvable : {e}"); return
    except Exception as e:
        st.error(f"Erreur : {e}"); return

    c1, c2, c3 = st.columns(3)
    c1.metric("Foyers", f"{res['n_pdl']}")
    c2.metric("RP", f"{res['n_rp']}")
    c3.metric("RS", f"{res['n_rs']}")

    tab1, tab2, tab3 = st.tabs(["Approche", "Modèles", "Résultats"])

    # ───────────────── TAB 1 ─────────────────
    with tab1:
        st.markdown("### Pourquoi générer des courbes ?")
        st.markdown(
            "Les données de consommation sont soumises au RGPD. Générer "
            "des profils synthétiques permet de partager des données "
            "réalistes sans compromettre la vie privée, et d'augmenter "
            "un jeu d'entraînement quand les données réelles sont rares."
        )

        st.markdown("### Le principe du VAE")
        st.markdown(
            "Un auto-encodeur compresse une courbe en un petit vecteur "
            "(l'espace latent) puis la reconstruit. Le VAE ajoute une "
            "contrainte : l'espace latent doit ressembler à une gaussienne. "
            "Grâce à ça, on peut tirer un vecteur au hasard dans cet "
            "espace et le décoder pour obtenir une courbe plausible."
        )

        st.markdown("### Conditionnement RP / RS")
        st.markdown(
            "Le label (0 = RP, 1 = RS) est concaténé à l'entrée de "
            "l'encodeur et du décodeur. Le modèle apprend à séparer "
            "les deux types dans l'espace latent. À la génération, on "
            "choisit le label et on obtient une courbe du type voulu."
        )

        st.markdown("---")
        st.markdown("### Les données")
        st.markdown(
            f"Chaque foyer est représenté par son profil annuel : "
            f"**{res['n_days']} jours** de consommation quotidienne (kWh). "
            f"Les profils sont normalisés (centrés-réduits) avant "
            f"d'entrer dans le modèle."
        )

    # ───────────────── TAB 2 ─────────────────
    with tab2:
        st.markdown("### Deux architectures comparées")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Linéaire")
            st.markdown(
                "Couches denses classiques. L'encodeur prend les "
                f"{res['n_days']} valeurs + le label, les compresse via "
                "deux couches (256, 128) jusqu'à un espace latent de "
                "16 dimensions. Le décodeur fait le chemin inverse."
            )
            st.markdown(
                "Rapide à entraîner. Capte le niveau global et la "
                "saisonnalité mais a tendance à lisser les détails."
            )
        with col_b:
            st.markdown("#### Conv-Attention")
            st.markdown(
                "Deux convolutions 1D (stride 2) réduisent le signal "
                "de 364 à 91 pas de temps, puis un Transformer (4 têtes "
                "d'attention) capture les dépendances longues (hiver vs été). "
                "Le décodeur fait l'inverse : Transformer puis convolutions "
                "transposées."
            )
            st.markdown(
                "Plus lent à entraîner mais produit des courbes plus "
                "réalistes : les motifs hebdomadaires et saisonniers "
                "sont mieux respectés."
            )

        for nom, r in res["resultats"].items():
            tot = sum(p.numel() for p in r["model"].parameters())
            st.caption(f"{nom} : {tot:,} paramètres")

        st.markdown("#### Entraînement")
        st.plotly_chart(fig_training(res["hist_lin"], res["hist_conv"]),
                        use_container_width=True)

    # ───────────────── TAB 3 ─────────────────
    with tab3:
        st.markdown("### Courbes générées vs réelles")
        modele = st.radio("Modèle", list(res["resultats"].keys()),
                          horizontal=True)
        r = res["resultats"][modele]

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                fig_courbes(res["reels_rp"], r["gen_rp"],
                            "RP", C["rp"], C["gen"]),
                use_container_width=True)
        with col2:
            st.plotly_chart(
                fig_courbes(res["reels_rs"], r["gen_rs"],
                            "RS", C["rs"], C["gen"]),
                use_container_width=True)

        st.markdown("### Comparaison statistique")
        def table(comp):
            rv, gv = comp["reel"], comp["genere"]
            return pd.DataFrame({
                "Grandeur": ["Moyenne (kWh/j)", "Écart-type",
                             "Moyenne hiver", "Moyenne été"],
                "Réel": [f"{rv['moy_jour']:.1f}", f"{rv['std_jour']:.1f}",
                         f"{rv['moy_hiver']:.1f}", f"{rv['moy_ete']:.1f}"],
                "Généré": [f"{gv['moy_jour']:.1f}", f"{gv['std_jour']:.1f}",
                           f"{gv['moy_hiver']:.1f}", f"{gv['moy_ete']:.1f}"],
            })

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RP**")
            st.dataframe(table(r["comp_rp"]), use_container_width=True,
                         hide_index=True)
        with col2:
            st.markdown("**RS**")
            st.dataframe(table(r["comp_rs"]), use_container_width=True,
                         hide_index=True)

        st.markdown("---")
        st.markdown("### Génération à la demande")
        col_ctrl = st.columns(3)
        type_res = col_ctrl[0].selectbox("Type", ["RP", "RS"])
        n_gen = col_ctrl[1].slider("Nombre", 1, 30, 5)

        if col_ctrl[2].button("Générer"):
            label_val = 0 if type_res == "RP" else 1
            nouvelles = generer(r["model"], label_val, n_gen, res["stats"])
            fig = go.Figure()
            for i in range(n_gen):
                fig.add_trace(go.Scatter(y=nouvelles[i], mode="lines",
                    line=dict(width=1.5), name=f"#{i+1}", showlegend=False))
            fig.update_layout(
                xaxis_title="Jour", yaxis_title="kWh/jour",
                height=320, margin=dict(l=50, r=20, t=10, b=40))
            st.plotly_chart(fig, use_container_width=True)