"""
Module Génération de courbes de charge — VAE conditionnel
================================================================

OBJECTIF
Générer des profils de consommation annuels synthétiques conditionnés
au type de résidence (RP ou RS). Un profil annuel = 364 jours de
consommation quotidienne en kWh (52 semaines complètes).

DEUX MODÈLES COMPARÉS
  Linéaire        : couches denses classiques, rapide à entraîner.
                    Capte le niveau global mais lisse les détails.
  Conv-Attention  : convolutions 1D + Transformer. Les convolutions
                    captent les motifs locaux (semaine), le Transformer
                    capte les dépendances longues (saison). Produit des
                    courbes plus réalistes.

Les deux sont conditionnels (CVAE) : le label RP/RS est fourni à
l'encodeur et au décodeur pour que le modèle associe chaque type
à une zone différente de l'espace latent.

PIPELINE
  1. charger_donnees / charger_labels
  2. construire_profils_annuels : une ligne par PDL, 364 colonnes
  3. LinearCVAE / ConvAttentionCVAE
  4. entrainer (les deux modèles)
  5. generer
  6. comparer (réel vs généré)
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


N_DAYS = 364      # 52 semaines complètes
LATENT_DIM = 16


# ================================================================
# 1. CHARGEMENT
# ================================================================

def charger_donnees(csv_path: str) -> pd.DataFrame:
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
    cols = list(lab.columns)
    lab = lab.rename(columns={cols[0]: "ID", cols[-1]: "label_raw"})
    lab["ID"] = lab["ID"].astype(str).str.strip()
    raw = lab["label_raw"].astype(str).str.strip().str.upper()
    mapping = {"RP": 0, "RS": 1, "P": 0, "S": 1, "0": 0, "1": 1}
    lab["label"] = raw.map(mapping)
    lab = lab.dropna(subset=["label"])
    lab["label"] = lab["label"].astype(int)
    return lab[["ID", "label"]]


# ================================================================
# 2. PROFILS ANNUELS
# ================================================================

def construire_profils_annuels(df, labels, n_days=N_DAYS):
    """
    Construit un profil annuel par PDL : vecteur de n_days valeurs
    de consommation journalière (kWh). Ne garde que les PDL qui ont
    assez de jours ET qui sont présents dans le fichier de labels.
    """
    df = df.copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df["date"] = df["horodate"].dt.date
    df["kwh"] = df["valeur"] * 0.5 / 1000

    daily = df.groupby(["ID", "date"])["kwh"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    labels_dict = dict(zip(labels["ID"].astype(str), labels["label"]))

    profils, labs = [], []
    for pdl_id, grp in daily.groupby("ID"):
        pdl_str = str(pdl_id).strip()
        if pdl_str not in labels_dict:
            continue
        grp = grp.sort_values("date")
        vals = grp["kwh"].values
        if len(vals) < n_days:
            continue
        profils.append(vals[:n_days])
        labs.append(labels_dict[pdl_str])

    profils = np.array(profils, dtype=np.float32)    # (n_pdl, n_days)
    labs = np.array(labs, dtype=np.int64)

    # Normalisation globale
    p_mean, p_std = profils.mean(), profils.std() + 1e-8
    profils_norm = (profils - p_mean) / p_std

    return profils_norm, labs, profils, {"mean": p_mean, "std": p_std}


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
# 4. MODÈLE 1 : LINÉAIRE (CVAE)
# ================================================================

class LinearCVAE(nn.Module):
    """VAE conditionnel à couches denses. Simple et rapide.
    Le label est concaténé à l'entrée de l'encodeur et du décodeur."""

    def __init__(self, input_dim=N_DAYS, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x, label):
        h = self.encoder(torch.cat([x, label.unsqueeze(1)], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, label):
        return self.decoder(torch.cat([z, label.unsqueeze(1)], dim=1))

    def forward(self, x, label):
        mu, logvar = self.encode(x, label)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, label), mu, logvar


# ================================================================
# 5. MODÈLE 2 : CONV-ATTENTION (CVAE)
# ================================================================

class ConvAttentionCVAE(nn.Module):
    """
    VAE conditionnel combinant convolutions 1D et Transformer.
    Les convolutions captent les motifs courts (rythme hebdomadaire).
    Le Transformer capte les dépendances longues (saisonnalité).
    """

    def __init__(self, input_dim=N_DAYS, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Taille après deux convolutions stride 2 : input_dim / 4
        self.seq_len = input_dim // 4    # 364 → 91
        self.d_model = 32

        # Encodeur : Conv1D → Transformer → latent
        self.cnn_enc = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(16, self.d_model, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, self.seq_len, self.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=1)

        flat_dim = self.seq_len * self.d_model   # 91 × 32 = 2912
        self.fc_mu = nn.Linear(flat_dim + 1, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim + 1, latent_dim)

        # Décodeur : latent → Transformer → ConvTranspose1D
        self.fc_z = nn.Linear(latent_dim + 1, flat_dim)
        self.pos_dec = nn.Parameter(torch.randn(1, self.seq_len, self.d_model))
        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer_dec = nn.TransformerEncoder(dec_layer, num_layers=1)
        self.cnn_dec = nn.Sequential(
            nn.ConvTranspose1d(self.d_model, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x, label):
        # x : (batch, input_dim) → (batch, 1, input_dim) pour Conv1d
        h = self.cnn_enc(x.unsqueeze(1))                # (batch, 32, seq_len)
        h = h.permute(0, 2, 1) + self.pos_enc           # (batch, seq_len, 32)
        h = self.transformer_enc(h)
        h_flat = h.reshape(x.size(0), -1)               # (batch, 2912)
        h_cat = torch.cat([h_flat, label.unsqueeze(1)], dim=1)
        return self.fc_mu(h_cat), self.fc_logvar(h_cat)

    def decode(self, z, label):
        h = self.fc_z(torch.cat([z, label.unsqueeze(1)], dim=1))
        h = h.reshape(-1, self.seq_len, self.d_model) + self.pos_dec
        h = self.transformer_dec(h)
        h = h.permute(0, 2, 1)                          # (batch, 32, seq_len)
        out = self.cnn_dec(h)                            # (batch, 1, ~input_dim)
        return out.squeeze(1)[:, :self.input_dim]        # ajuster si taille ≠

    def forward(self, x, label):
        mu, logvar = self.encode(x, label)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, label), mu, logvar


# ================================================================
# 6. PERTE
# ================================================================

def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


# ================================================================
# 7. ENTRAÎNEMENT
# ================================================================

def entrainer(model, profils, labels, epochs=200, batch_size=32,
              lr=1e-3, beta=0.05, patience=20):
    n = len(profils)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    n_tr = int(0.85 * n)
    train_ds = ProfilDataset(profils[idx[:n_tr]], labels[idx[:n_tr]])
    val_ds = ProfilDataset(profils[idx[n_tr:]], labels[idx[n_tr:]])
    train_ld = DataLoader(train_ds, batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"train": [], "val": []}
    best_val, best_state, no_imp = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        tl = 0
        for xb, yb in train_ld:
            recon, mu, logvar = model(xb, yb)
            loss, _, _ = vae_loss(recon, xb, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tl += loss.item()
        hist["train"].append(tl / len(train_ld))

        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in val_ld:
                recon, mu, logvar = model(xb, yb)
                loss, _, _ = vae_loss(recon, xb, mu, logvar, beta)
                vl += loss.item()
        hist["val"].append(vl / len(val_ld))

        if hist["val"][-1] < best_val:
            best_val = hist["val"][-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    hist["stopped"] = epoch + 1
    return hist


# ================================================================
# 8. GÉNÉRATION
# ================================================================

def generer(model, label, n=20, stats=None):
    """Tire n profils annuels pour le type voulu (0=RP, 1=RS)."""
    model.eval()
    z = torch.randn(n, model.latent_dim)
    lab = torch.full((n,), float(label))
    with torch.no_grad():
        gen = model.decode(z, lab).numpy()
    if stats:
        gen = gen * stats["std"] + stats["mean"]
        gen = np.maximum(gen, 0)
    return gen


# ================================================================
# 9. COMPARAISON
# ================================================================

def comparer(reels, generes):
    def stats(arr):
        weekly = arr.reshape(arr.shape[0], -1, 7).mean(axis=2)
        return {
            "moy_jour": float(arr.mean()),
            "std_jour": float(arr.std()),
            "moy_hiver": float(arr[:, :90].mean()),    # ~jan-mars
            "moy_ete": float(arr[:, 180:270].mean()),   # ~juil-sept
        }
    return {"reel": stats(reels), "genere": stats(generes)}


# ================================================================
# 10. PIPELINE
# ================================================================

def pipeline_complet(csv_path, labels_path, epochs=200, batch_size=32,
                     lr=1e-3, beta=0.05):
    df = charger_donnees(csv_path)
    labels = charger_labels(labels_path)
    profils_norm, labs, profils_bruts, stats = construire_profils_annuels(df, labels)

    input_dim = profils_norm.shape[1]

    # Entraîner les deux modèles
    model_lin = LinearCVAE(input_dim=input_dim)
    hist_lin = entrainer(model_lin, profils_norm, labs,
                         epochs=epochs, batch_size=batch_size, lr=lr, beta=beta)

    model_conv = ConvAttentionCVAE(input_dim=input_dim)
    hist_conv = entrainer(model_conv, profils_norm, labs,
                          epochs=epochs, batch_size=batch_size, lr=lr, beta=beta)

    # Générer pour chaque modèle et chaque type
    mask_rp, mask_rs = labs == 0, labs == 1
    resultats = {}
    for nom, model in [("Linéaire", model_lin), ("Conv-Attention", model_conv)]:
        gen_rp = generer(model, 0, n=50, stats=stats)
        gen_rs = generer(model, 1, n=50, stats=stats)
        resultats[nom] = {
            "model": model,
            "gen_rp": gen_rp, "gen_rs": gen_rs,
            "comp_rp": comparer(profils_bruts[mask_rp], gen_rp),
            "comp_rs": comparer(profils_bruts[mask_rs], gen_rs),
        }

    return {
        "resultats": resultats,
        "hist_lin": hist_lin,
        "hist_conv": hist_conv,
        "reels_rp": profils_bruts[mask_rp],
        "reels_rs": profils_bruts[mask_rs],
        "stats": stats,
        "n_pdl": len(profils_norm),
        "n_rp": int(mask_rp.sum()),
        "n_rs": int(mask_rs.sum()),
        "n_days": input_dim,
    }