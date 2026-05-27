"""
Module Génération de courbes de charge — VAE conditionnel
================================================================

OBJECTIF
Générer des courbes de consommation synthétiques qui ressemblent aux
courbes réelles, en choisissant le type de résidence (RP ou RS).

STRATÉGIE
On utilise un auto-encodeur variationnel conditionnel (CVAE). Le principe
est en trois temps :
  1. L'encodeur compresse une courbe réelle en un petit vecteur (l'espace
     latent), en lui associant le label RP ou RS.
  2. Le décodeur reconstruit une courbe à partir de ce vecteur + le label.
  3. À la génération, on tire un vecteur au hasard dans l'espace latent
     et on le décode avec le label voulu. Le résultat est une courbe
     synthétique qui a les propriétés statistiques d'une RP ou d'une RS.

DONNÉES
On travaille sur des profils journaliers : 48 mesures de puissance (W)
par jour (pas de 30 min). C'est le grain le plus parlant pour distinguer
RP (double pic matin/soir) et RS (consommation basse ou épisodique).

ÉTAPES DU PIPELINE
  1. charger_donnees           : lecture du CSV brut
  2. charger_labels            : lecture du fichier de labels RP/RS
  3. extraire_profils          : pivote en matrice (n_profils, 48)
  4. CVAE                      : le modèle (encodeur + décodeur)
  5. entrainer                 : apprentissage sur les courbes réelles
  6. generer                   : tirage de courbes synthétiques
  7. comparer_distributions    : métriques réel vs généré
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ================================================================
# 1. CHARGEMENT DES DONNÉES
# ================================================================

def charger_donnees(csv_path: str) -> pd.DataFrame:
    """Charge le CSV brut (ID, horodate, valeur). Gère les séparateurs
    courants et les fuseaux horaires."""
    for sep in [",", ";"]:
        for enc in ["utf-8", "latin-1"]:
            try:
                df = pd.read_csv(csv_path, sep=sep, encoding=enc, nrows=5)
                if len(df.columns) >= 3:
                    df = pd.read_csv(csv_path, sep=sep, encoding=enc)
                    break
            except Exception:
                continue
        else:
            continue
        break
    else:
        raise ValueError(f"Impossible de lire {csv_path}.")

    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("id", "pdl", "identifiant"):
            col_map[c] = "ID"
        elif cl in ("horodate", "date", "datetime", "timestamp"):
            col_map[c] = "horodate"
        elif cl in ("valeur", "value", "puissance", "conso", "w", "watts"):
            col_map[c] = "valeur"
    if len(col_map) < 3 and len(df.columns) >= 3:
        oc = list(df.columns)
        for name, pos in [("ID", 0), ("horodate", 1), ("valeur", 2)]:
            if name not in col_map.values():
                col_map[oc[pos]] = name
    df = df.rename(columns=col_map)

    df["horodate"] = pd.to_datetime(df["horodate"], utc=True)
    df["horodate"] = df["horodate"].dt.tz_convert("Europe/Paris")
    df["valeur"] = pd.to_numeric(df["valeur"], errors="coerce")
    df = df.dropna(subset=["valeur"])
    return df


def charger_labels(labels_path: str) -> pd.DataFrame:
    """Charge le fichier de labels (ID, label). Le label peut être
    RP/RS, 0/1, ou P/S selon le format du fichier."""
    for sep in [",", ";"]:
        for enc in ["utf-8", "latin-1"]:
            try:
                lab = pd.read_csv(labels_path, sep=sep, encoding=enc)
                if len(lab.columns) >= 2:
                    break
            except Exception:
                continue
        else:
            continue
        break
    else:
        raise ValueError(f"Impossible de lire {labels_path}.")

    lab.columns = lab.columns.str.strip().str.replace('"', '')
    # Identifier la colonne ID et la colonne label
    cols = list(lab.columns)
    lab = lab.rename(columns={cols[0]: "ID", cols[-1]: "label_raw"})
    lab["ID"] = lab["ID"].astype(str).str.strip()

    # Convertir en 0 (RP) / 1 (RS)
    raw = lab["label_raw"].astype(str).str.strip().str.upper()
    mapping = {"RP": 0, "RS": 1, "P": 0, "S": 1, "0": 0, "1": 1}
    lab["label"] = raw.map(mapping)
    lab = lab.dropna(subset=["label"])
    lab["label"] = lab["label"].astype(int)
    return lab[["ID", "label"]]


# ================================================================
# 2. EXTRACTION DES PROFILS JOURNALIERS
# ================================================================

def extraire_profils(df: pd.DataFrame, labels: pd.DataFrame, min_jours: int = 30):
    """
    Transforme les mesures 30 min en profils journaliers (48 valeurs).
    Ne garde que les jours complets (48 mesures) et les PDL présents
    dans le fichier de labels.

    Normalisation par PDL : chaque foyer est ramené à sa propre échelle
    (division par sa puissance moyenne). Le modèle apprend des formes
    de courbe, pas des niveaux absolus.

    Retourne :
        profils : array (n, 48) normalisé
        labels  : array (n,) avec 0=RP, 1=RS
        stats   : dict {pdl_id: (mean, std)} pour dénormaliser
    """
    df = df.copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df["date"] = df["horodate"].dt.date

    labels_dict = dict(zip(labels["ID"].astype(str), labels["label"]))

    profils_list, labels_list, stats = [], [], {}

    for pdl_id, grp in df.groupby("ID"):
        pdl_str = str(pdl_id).strip()
        if pdl_str not in labels_dict:
            continue
        label = labels_dict[pdl_str]

        # Pivoter par jour : chaque ligne = 1 jour, 48 colonnes
        grp = grp.sort_values("horodate")
        for date, day_grp in grp.groupby("date"):
            vals = day_grp["valeur"].values
            if len(vals) != 48:          # jour incomplet, on saute
                continue
            profils_list.append(vals)
            labels_list.append(label)

        # Stats du PDL pour dénormaliser plus tard
        all_vals = grp["valeur"].values
        stats[pdl_str] = (float(all_vals.mean()), float(all_vals.std() + 1e-8))

    profils = np.array(profils_list, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.int64)

    # Normalisation globale (centrer-réduire sur l'ensemble)
    p_mean = profils.mean()
    p_std = profils.std() + 1e-8
    profils_norm = (profils - p_mean) / p_std

    return profils_norm, labels_arr, profils, {"mean": p_mean, "std": p_std}


# ================================================================
# 3. DATASET
# ================================================================

class ProfilDataset(Dataset):
    def __init__(self, profils, labels):
        self.X = torch.tensor(profils, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ================================================================
# 4. CVAE (Conditional Variational Auto-Encoder)
# ================================================================

LATENT_DIM = 8   # taille de l'espace latent
INPUT_DIM = 48   # 48 pas de 30 min par jour


class CVAE(nn.Module):
    """
    Auto-encodeur variationnel conditionnel.

    L'encodeur prend un profil de 48 valeurs + le label (49 entrées)
    et produit deux vecteurs de taille LATENT_DIM : la moyenne (mu)
    et le log de la variance (logvar) de la distribution latente.

    Le décodeur prend un point de l'espace latent + le label et
    reconstruit un profil de 48 valeurs.

    Le conditionnement (label RP/RS) est simplement concaténé à l'entrée
    de l'encodeur et du décodeur. C'est la manière la plus directe
    d'obtenir un modèle conditionnel.
    """

    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Encodeur : profil (48) + label (1) → espace latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Décodeur : latent (8) + label (1) → profil reconstruit (48)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def encode(self, x, label):
        # Concaténer le profil et le label
        h = self.encoder(torch.cat([x, label.unsqueeze(1)], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu, logvar):
        # Astuce de reparamétrisation : z = mu + sigma * epsilon
        # Permet de rétro-propager le gradient à travers l'échantillonnage
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, label):
        return self.decoder(torch.cat([z, label.unsqueeze(1)], dim=1))

    def forward(self, x, label):
        mu, logvar = self.encode(x, label)
        z = self.reparametrize(mu, logvar)
        recon = self.decode(z, label)
        return recon, mu, logvar


# ================================================================
# 5. FONCTION DE PERTE
# ================================================================

def vae_loss(recon, x, mu, logvar, beta=1.0):
    """
    Deux termes additionnés :
      reconstruction : écart entre la courbe originale et la courbe reconstruite (MSE)
      KL divergence  : force l'espace latent à rester proche d'une gaussienne N(0,1)
                       (c'est ce qui permet de générer en tirant z au hasard)
    beta contrôle l'équilibre entre les deux. beta=1 donne le VAE standard.
    """
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


# ================================================================
# 6. ENTRAÎNEMENT
# ================================================================

def entrainer(profils, labels, epochs=120, batch_size=64, lr=1e-3,
              beta=1.0, patience=15):
    """Entraîne le CVAE avec early stopping sur un split 85/15."""
    n = len(profils)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    n_train = int(0.85 * n)
    train_ds = ProfilDataset(profils[idx[:n_train]], labels[idx[:n_train]])
    val_ds = ProfilDataset(profils[idx[n_train:]], labels[idx[n_train:]])
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)

    model = CVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"train": [], "val": [], "recon": [], "kl": []}
    best_val, best_state, no_imp = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for xb, yb in train_loader:
            recon, mu, logvar = model(xb, yb)
            loss, _, _ = vae_loss(recon, xb, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        model.eval()
        v_loss, v_recon, v_kl = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                recon, mu, logvar = model(xb, yb)
                loss, rl, kl = vae_loss(recon, xb, mu, logvar, beta)
                v_loss += loss.item()
                v_recon += rl
                v_kl += kl
        n_batches = len(val_loader)
        v_loss /= n_batches

        hist["train"].append(t_loss)
        hist["val"].append(v_loss)
        hist["recon"].append(v_recon / n_batches)
        hist["kl"].append(v_kl / n_batches)

        if v_loss < best_val:
            best_val, no_imp = v_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    hist["stopped_epoch"] = epoch + 1
    return model, hist


# ================================================================
# 7. GÉNÉRATION
# ================================================================

def generer(model, label, n=50, stats=None):
    """
    Génère n profils journaliers pour le type demandé (0=RP, 1=RS).
    Tire z au hasard dans N(0,1) et décode avec le label voulu.
    Si stats est fourni, dénormalise en watts.
    """
    model.eval()
    z = torch.randn(n, model.latent_dim)
    lab = torch.full((n,), float(label))
    with torch.no_grad():
        gen = model.decode(z, lab).numpy()
    if stats:
        gen = gen * stats["std"] + stats["mean"]
        gen = np.maximum(gen, 0)     # la puissance ne peut pas être négative
    return gen


# ================================================================
# 8. COMPARAISON RÉEL / GÉNÉRÉ
# ================================================================

def comparer(reels, generes):
    """
    Quelques métriques simples pour vérifier que les courbes générées
    ont les mêmes propriétés statistiques que les courbes réelles.
    """
    def stats(arr):
        return {
            "moyenne": float(arr.mean()),
            "ecart_type": float(arr.std()),
            "pic_matin": float(arr[:, 14:20].mean()),     # 7h-10h
            "pic_soir": float(arr[:, 36:44].mean()),       # 18h-22h
            "creux_nuit": float(arr[:, 0:8].mean()),       # 0h-4h
        }

    return {"reel": stats(reels), "genere": stats(generes)}


# ================================================================
# 9. PIPELINE COMPLET
# ================================================================

def pipeline_complet(csv_path, labels_path, epochs=120, batch_size=64,
                     lr=1e-3, beta=1.0):
    df = charger_donnees(csv_path)
    labels = charger_labels(labels_path)
    profils_norm, labels_arr, profils_bruts, stats = extraire_profils(df, labels)

    model, hist = entrainer(profils_norm, labels_arr, epochs=epochs,
                            batch_size=batch_size, lr=lr, beta=beta)

    # Générer des exemples pour chaque type
    gen_rp = generer(model, label=0, n=100, stats=stats)
    gen_rs = generer(model, label=1, n=100, stats=stats)

    # Séparer les profils réels par type
    mask_rp = labels_arr == 0
    mask_rs = labels_arr == 1
    reels_rp = profils_bruts[mask_rp]
    reels_rs = profils_bruts[mask_rs]

    comp_rp = comparer(reels_rp, gen_rp)
    comp_rs = comparer(reels_rs, gen_rs)

    return {
        "model": model,
        "historique": hist,
        "gen_rp": gen_rp,
        "gen_rs": gen_rs,
        "reels_rp": reels_rp,
        "reels_rs": reels_rs,
        "comp_rp": comp_rp,
        "comp_rs": comp_rs,
        "stats": stats,
        "n_profils": len(profils_norm),
        "n_rp": int(mask_rp.sum()),
        "n_rs": int(mask_rs.sum()),
    }