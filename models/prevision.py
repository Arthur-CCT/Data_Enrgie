"""
Module Prévision de consommation
================================================================

IDÉE PRINCIPALE
Prédire jour par jour la conso d'un foyer est presque impossible :
un jour donné dépend de comportements imprévisibles (sorties, invités…).
On sépare donc la prévision en deux morceaux :

  NIVEAU  = combien d'énergie sur la semaine entière (1 nombre, kWh).
            En sommant les 7 jours, le hasard de chaque jour s'annule
            en partie : c'est BEAUCOUP plus prévisible.
  FORME   = comment cette énergie se répartit sur les 7 jours.
            C'est l'habitude du foyer (creux en semaine, pic le week-end),
            très stable d'une semaine à l'autre.

  Prévision d'un jour  =  NIVEAU prédit  ×  part de ce jour dans la FORME

Le CNN-LSTM ne prédit que le NIVEAU. La FORME est calculée directement
à partir de l'historique du foyer.

ÉTAPES DU PIPELINE
  1. charger_donnees        : lecture du CSV brut
  2. agreger_journalier     : puissance 30 min  ->  énergie par jour (kWh)
  3. construire_donnees_pdl : une matrice (semaines × 7 jours) par foyer
  4. creer_fenetres         : découpe en exemples + calcule NIVEAU et FORME
  5. CNNLSTM / entrainer    : le modèle apprend le NIVEAU
  6. predire                : NIVEAU × FORME  ->  prévision des 7 jours
  7. metriques              : comparaison au réel et aux méthodes naïves
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ═════════════════════════════════════════════════════════════════

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
        if cl in ("id", "pdl", "identifiant", "id_pdl"):
            col_map[c] = "ID"
        elif cl in ("horodate", "date", "datetime", "timestamp", "heure"):
            col_map[c] = "horodate"
        elif cl in ("valeur", "value", "puissance", "conso", "w", "watts"):
            col_map[c] = "valeur"
    if len(col_map) < 3 and len(df.columns) >= 3:
        oc = list(df.columns)
        for name, pos in [("ID", 0), ("horodate", 1), ("valeur", 2)]:
            if name not in col_map.values():
                col_map[oc[pos]] = name
    df = df.rename(columns=col_map)
    for needed in ["ID", "horodate", "valeur"]:
        if needed not in df.columns:
            raise KeyError(
                f"Colonne '{needed}' introuvable. "
                f"Colonnes détectées : {list(df.columns)}"
            )
    df["horodate"] = pd.to_datetime(df["horodate"], utc=True)
    df["horodate"] = df["horodate"].dt.tz_convert("Europe/Paris")
    df["valeur"] = pd.to_numeric(df["valeur"], errors="coerce")
    df = df.dropna(subset=["valeur"])
    return df


def agreger_journalier(df: pd.DataFrame) -> pd.DataFrame:
    """Puissance (W) au pas 30 min  ->  énergie consommée par jour (kWh).
    Énergie = puissance × durée : W × 0.5 h = Wh, /1000 = kWh, puis somme du jour."""
    df = df.copy()
    df["date"] = df["horodate"].dt.date
    df["energie_kwh"] = df["valeur"] * 0.5 / 1000
    daily = (
        df.groupby(["ID", "date"])["energie_kwh"].sum()
        .reset_index().rename(columns={"energie_kwh": "conso_kwh"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# ═════════════════════════════════════════════════════════════════
# 2. DONNÉES PAR PDL
# ═════════════════════════════════════════════════════════════════

def construire_donnees_pdl(daily: pd.DataFrame, min_semaines: int = 12):
    """Range la conso de chaque foyer en matrice (semaines × 7 jours).
    On garde aussi sa moyenne et son écart-type : ils servent à mettre
    tous les foyers à la même échelle (un petit studio et une grande
    maison auront alors des courbes comparables)."""
    pdl_data = {}
    for pdl_id, grp in daily.groupby("ID"):
        grp = grp.sort_values("date").reset_index(drop=True)
        vals = grp["conso_kwh"].values
        dates = grp["date"].values
        n_jours = (len(vals) // 7) * 7          # tronque à un nombre entier de semaines
        if n_jours < min_semaines * 7:
            continue
        vals = vals[:n_jours]
        dates = dates[:n_jours]
        pdl_data[pdl_id] = {
            "conso_raw": vals.reshape(-1, 7),   # (semaines, 7)
            "mean": vals.mean(),
            "std": vals.std() + 1e-8,
            "dates": dates.reshape(-1, 7),
        }
    return pdl_data


# ═════════════════════════════════════════════════════════════════
# 3. FEATURES TEMPORELLES
# ═════════════════════════════════════════════════════════════════

def encoder_temporel(dates_matrix):
    """Donne au modèle un repère de saison via sin/cos du jour de l'année.
    sin et cos (plutôt qu'un simple numéro de jour) évitent la coupure
    entre le 31 déc et le 1er jan : décembre et janvier restent proches."""
    n_sem = dates_matrix.shape[0]
    sin_vals = np.zeros(n_sem)
    cos_vals = np.zeros(n_sem)
    for i in range(n_sem):
        dt = pd.Timestamp(dates_matrix[i, 3])           # milieu de semaine
        angle = 2 * np.pi * dt.day_of_year / 365.25
        sin_vals[i] = np.sin(angle)
        cos_vals[i] = np.cos(angle)
    return sin_vals, cos_vals


# ═════════════════════════════════════════════════════════════════
# 4. FENÊTRAGE : DÉCOMPOSITION NIVEAU × FORME
# ═════════════════════════════════════════════════════════════════

def creer_fenetres(pdl_data, horizon=8, test_ratio=0.2, val_ratio=0.15):
    """
    Découpe chaque foyer en exemples par fenêtre glissante : on regarde
    H semaines pour prédire la semaine suivante.

    Pour chaque exemple on prépare :
      - X        : l'historique (H semaines × 7 jours) + sin/cos, pour le modèle
      - baseline : NIVEAU "naïf" = moyenne des totaux des H semaines
                   (les semaines récentes comptent un peu plus)
      - résidu   : ce que le modèle doit prédire = vrai total - baseline
                   (apprendre une petite correction est plus facile que
                    de repartir de zéro)
      - forme    : part de chaque jour dans la semaine (somme = 1)

    Découpage train/val/test fait dans l'ordre du temps (jamais au hasard),
    pour ne pas entraîner le modèle sur des semaines postérieures au test.
    """
    # Poids de récence : la dernière semaine pèse 2× plus que la première
    w = np.linspace(1.0, 2.0, horizon)
    w = w / w.sum()

    out = {k: [] for k in [
        "X_train", "res_train", "X_val", "res_val", "X_test", "res_test",
        "base_test", "shape_test", "ytot_test", "yday_test",
        "X_test_raw", "meta_test", "persist_test",
    ]}

    for pdl_id, data in pdl_data.items():
        raw = data["conso_raw"]          # (n_sem, 7) kWh
        mean, std = data["mean"], data["std"]
        norm = (raw - mean) / std
        dates = data["dates"]
        n_sem = raw.shape[0]
        if n_sem <= horizon:
            continue
        sin_vals, cos_vals = encoder_temporel(dates)

        rows = []
        for i in range(n_sem - horizon):
            hist_raw = raw[i: i + horizon]            # (H, 7)
            hist_norm = norm[i: i + horizon]
            sin_col = sin_vals[i: i + horizon, None]
            cos_col = cos_vals[i: i + horizon, None]
            x_row = np.concatenate([hist_norm, sin_col, cos_col], axis=1)

            wk_tot = hist_raw.sum(axis=1)             # (H,) totaux hebdo
            baseline = float((wk_tot * w).sum())      # moy pondérée récence

            shape = (hist_raw * w[:, None]).sum(axis=0)  # (7,) pondérée
            shape = shape / (shape.sum() + 1e-8)

            y_next = raw[i + horizon]                 # (7,) kWh réels
            y_tot = float(y_next.sum())
            residual = y_tot - baseline               # cible du modèle

            persist = hist_raw[-1]                    # dernière semaine

            rows.append((x_row, residual, baseline, shape,
                         y_tot, y_next, hist_raw, (pdl_id, i + horizon), persist))

        n = len(rows)
        if n < 3:
            continue
        n_te = max(1, int(n * test_ratio))
        n_va = max(1, int(n * val_ratio))
        n_tr = n - n_te - n_va
        if n_tr < 1:
            continue

        for k, (x, r, b, sh, yt, yd, hr, meta, pe) in enumerate(rows):
            if k < n_tr:
                out["X_train"].append(x); out["res_train"].append(r)
            elif k < n_tr + n_va:
                out["X_val"].append(x); out["res_val"].append(r)
            else:
                out["X_test"].append(x); out["res_test"].append(r)
                out["base_test"].append(b); out["shape_test"].append(sh)
                out["ytot_test"].append(yt); out["yday_test"].append(yd)
                out["X_test_raw"].append(hr); out["meta_test"].append(meta)
                out["persist_test"].append(pe)

    for k in out:
        if k != "meta_test":
            out[k] = np.array(out[k])
    return out


# ═════════════════════════════════════════════════════════════════
# 5. DATASET + MODÈLE
# ═════════════════════════════════════════════════════════════════

class ConsoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X[:, None, :, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class CNNLSTM(nn.Module):
    """
    Réseau qui prédit le NIVEAU de la semaine à venir (1 seul nombre).

    Entrée : (batch, 1, H, 9)  -> H semaines, chacune décrite par
             7 jours de conso + 2 repères de saison (sin, cos)
    Sortie : (batch, 1)        -> la correction à ajouter à la baseline

    - Le CNN repère des motifs dans une semaine (forme jour/week-end).
    - Le LSTM lit les semaines dans l'ordre et capte la tendance.
    """

    def __init__(self, input_width=9):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.drop = nn.Dropout(0.15)
        self.lstm = nn.LSTM(32 * input_width, 128, num_layers=2,
                            batch_first=True, dropout=0.15)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc2(F.relu(self.fc1(x)))


# ═════════════════════════════════════════════════════════════════
# 6. ENTRAÎNEMENT
# ═════════════════════════════════════════════════════════════════

def entrainer_modele(X_train, res_train, X_val, res_val,
                     epochs=100, batch_size=32, lr=5e-4, patience=15):
    # Normaliser le résidu (centré-réduit) pour stabiliser l'apprentissage
    rm, rs = res_train.mean(), res_train.std() + 1e-8
    ytr = (res_train - rm) / rs
    yva = (res_val - rm) / rs

    train_loader = DataLoader(ConsoDataset(X_train, ytr), batch_size, shuffle=True)
    val_loader = DataLoader(ConsoDataset(X_val, yva), batch_size)

    model = CNNLSTM()
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5, min_lr=1e-6)

    hist = {"train_loss": [], "val_loss": []}
    best, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        tl = 0
        for X_b, y_b in train_loader:
            pred = model(X_b)
            loss = criterion(pred, y_b[:, None])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        tl /= len(train_loader)

        model.eval()
        vl = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                vl += criterion(model(X_b), y_b[:, None]).item()
        vl /= len(val_loader)
        scheduler.step(vl)
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)

        if vl < best:
            best, no_improve = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    hist["stopped_epoch"] = epoch + 1
    return model, hist, (rm, rs)


# ═════════════════════════════════════════════════════════════════
# 7. PRÉDICTION (décomposition)
# ═════════════════════════════════════════════════════════════════

def predire(model, X_test, base_test, shape_test, res_stats):
    """Reconstruit la prévision des 7 jours :
        NIVEAU prédit = baseline + correction du modèle
        7 jours       = NIVEAU prédit × forme du foyer"""
    rm, rs = res_stats
    ds = ConsoDataset(X_test, np.zeros(len(X_test)))
    loader = DataLoader(ds, batch_size=64)
    model.eval()
    res = []
    with torch.no_grad():
        for X_b, _ in loader:
            res.append(model(X_b).cpu().numpy().ravel())
    res = np.concatenate(res) * rs + rm

    total_pred = np.maximum(base_test + res, 0)        # (n,)
    daily_pred = total_pred[:, None] * shape_test       # (n, 7)
    return total_pred, daily_pred


# ═════════════════════════════════════════════════════════════════
# 8. MÉTRIQUES
# ═════════════════════════════════════════════════════════════════

MAPE_SEUIL = 2.0


def mape(pred, cible, seuil=MAPE_SEUIL):
    m = cible > seuil
    return float(np.abs((pred[m] - cible[m]) / cible[m]).mean() * 100) if m.any() else np.nan


def metriques(pred, cible, nom=""):
    err = pred - cible
    return {
        "nom": nom,
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE": mape(pred, cible),
    }


def metriques_par_jour(pred, cible):
    jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    return {
        jours[j]: {
            "MAE": float(np.abs(pred[:, j] - cible[:, j]).mean()),
            "RMSE": float(np.sqrt(((pred[:, j] - cible[:, j]) ** 2).mean())),
        } for j in range(7)
    }


# ═════════════════════════════════════════════════════════════════
# 9. PIPELINE
# ═════════════════════════════════════════════════════════════════

def pipeline_complet(csv_path, horizon=8, epochs=100, batch_size=32,
                     lr=5e-4, test_ratio=0.2, val_ratio=0.15):
    df = charger_donnees(csv_path)
    daily = agreger_journalier(df)
    pdl_data = construire_donnees_pdl(daily, min_semaines=horizon + 4)
    sp = creer_fenetres(pdl_data, horizon, test_ratio, val_ratio)

    model, hist, res_stats = entrainer_modele(
        sp["X_train"], sp["res_train"], sp["X_val"], sp["res_val"],
        epochs=epochs, batch_size=batch_size, lr=lr)

    total_pred, daily_pred = predire(
        model, sp["X_test"], sp["base_test"], sp["shape_test"], res_stats)

    ytot = sp["ytot_test"]            # totaux hebdo réels
    yday = sp["yday_test"]            # 7 jours réels
    base = sp["base_test"]            # baseline (sans NN)
    persist = sp["persist_test"]      # persistence (dernière semaine)

    # baseline seule en journalier
    daily_base = base[:, None] * sp["shape_test"]

    return {
        "model": model, "historique": hist,
        # niveau (total hebdo)
        "total_pred": total_pred, "total_reel": ytot,
        "metr_total": metriques(total_pred, ytot, "CNN-LSTM"),
        "metr_total_base": metriques(base, ytot, "Baseline (moy. pond.)"),
        "metr_total_persist": metriques(persist.sum(1), ytot, "Persistence"),
        # journalier
        "daily_pred": daily_pred, "daily_reel": yday,
        "daily_base": daily_base, "daily_persist": persist,
        "metr_daily": metriques(daily_pred, yday, "CNN-LSTM"),
        "metr_daily_base": metriques(daily_base, yday, "Décomp. baseline"),
        "metr_daily_persist": metriques(persist, yday, "Persistence"),
        "metr_jours": metriques_par_jour(daily_pred, yday),
        # meta
        "shape_test": sp["shape_test"],
        "X_test_raw": sp["X_test_raw"], "meta_test": sp["meta_test"],
        "n_pdl": len(pdl_data),
        "n_train": len(sp["X_train"]), "n_test": len(sp["X_test"]),
        "n_val": len(sp["X_val"]),
        "horizon": horizon, "epochs": hist["stopped_epoch"],
    }