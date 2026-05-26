"""
Module Prévision de consommation (CNN-LSTM) :
==================================================
  1. Normalisation par PDL (et non globale)
  2. Features temporelles (sin/cos semaine de l'année) → saisonnalité
  3. Horizon par défaut étendu à 8 semaines
  4. Split temporel (dernières semaines = test, pas random)
  5. Early stopping sur la validation loss
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & PRÉTRAITEMENT
# ═════════════════════════════════════════════════════════════════

def charger_donnees(csv_path: str) -> pd.DataFrame:
    """
    Charge le CSV brut et normalise les colonnes vers (ID, horodate, valeur).
    Gère séparateurs , et ;, encodages utf-8 et latin-1, noms de colonnes
    variables, et timezones.
    """
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
        raise ValueError(
            f"Impossible de lire {csv_path}. "
            f"Vérifiez le format (CSV avec 3 colonnes : ID, horodate, valeur)."
        )

    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")

    col_map = {}
    cols_lower = {c: c.lower().strip() for c in df.columns}
    for c, cl in cols_lower.items():
        if cl in ("id", "pdl", "identifiant", "id_pdl"):
            col_map[c] = "ID"
        elif cl in ("horodate", "date", "datetime", "timestamp", "heure"):
            col_map[c] = "horodate"
        elif cl in ("valeur", "value", "puissance", "conso", "w", "watts"):
            col_map[c] = "valeur"

    if len(col_map) < 3 and len(df.columns) >= 3:
        original_cols = list(df.columns)
        if "ID" not in col_map.values():
            col_map[original_cols[0]] = "ID"
        if "horodate" not in col_map.values():
            col_map[original_cols[1]] = "horodate"
        if "valeur" not in col_map.values():
            col_map[original_cols[2]] = "valeur"

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
    """Agrège la puissance (W, pas 30 min) en énergie journalière (kWh)."""
    df = df.copy()
    df["date"] = df["horodate"].dt.date
    df["energie_kwh"] = df["valeur"] * 0.5 / 1000
    daily = (
        df.groupby(["ID", "date"])["energie_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"energie_kwh": "conso_kwh"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# ═════════════════════════════════════════════════════════════════
# 2. STRUCTURATION PAR PDL (normalisation individuelle + dates)
# ═════════════════════════════════════════════════════════════════

def construire_donnees_pdl(daily: pd.DataFrame, min_semaines: int = 10):
    """
    Pour chaque PDL :
      - matrice de conso brute et normalisée (n_semaines, 7)
      - matrice de dates (n_semaines, 7)
      - stats du PDL (mean, std) pour dénormaliser ensuite

    La normalisation par PDL est le changement clé : un studio à
    4 kWh/jour et un logement chauffage électrique à 20 kWh/jour
    sont ramenés à la même échelle. Le modèle apprend les *formes*
    de courbe, pas les niveaux absolus.
    """
    pdl_data = {}
    for pdl_id, grp in daily.groupby("ID"):
        grp = grp.sort_values("date").reset_index(drop=True)
        vals = grp["conso_kwh"].values
        dates = grp["date"].values

        n_jours = (len(vals) // 7) * 7
        if n_jours < min_semaines * 7:
            continue

        vals = vals[:n_jours]
        dates = dates[:n_jours]

        pdl_mean = vals.mean()
        pdl_std = vals.std() + 1e-8

        pdl_data[pdl_id] = {
            "conso_raw": vals.reshape(-1, 7),
            "conso_norm": ((vals - pdl_mean) / pdl_std).reshape(-1, 7),
            "dates": dates.reshape(-1, 7),
            "mean": pdl_mean,
            "std": pdl_std,
        }

    return pdl_data


# ═════════════════════════════════════════════════════════════════
# 3. FEATURES TEMPORELLES
# ═════════════════════════════════════════════════════════════════

def encoder_temporel(dates_matrix):
    """
    Encode la position dans l'année en sin/cos (1 valeur par semaine).

    Pourquoi ? Avec seulement 4-8 semaines d'historique, le modèle
    ne « voit » pas directement la saison. En ajoutant sin/cos du
    jour de l'année, on lui dit explicitement « on est en janvier »
    ou « on est en juillet » → il peut adapter sa prédiction à la
    saisonnalité (chauffage l'hiver, climatisation l'été).

    Pourquoi sin/cos et pas juste le numéro de semaine ?
    Le jour 1 (1er jan) et le jour 365 (31 déc) sont voisins dans
    le cycle annuel. Un encodage linéaire créerait une discontinuité.
    sin/cos assure la continuité circulaire.
    """
    n_sem = dates_matrix.shape[0]
    sin_vals = np.zeros(n_sem)
    cos_vals = np.zeros(n_sem)

    for i in range(n_sem):
        # Prendre le milieu de la semaine pour la position
        dt = pd.Timestamp(dates_matrix[i, 3])  # mercredi
        day_of_year = dt.day_of_year
        angle = 2 * np.pi * day_of_year / 365.25
        sin_vals[i] = np.sin(angle)
        cos_vals[i] = np.cos(angle)

    return sin_vals, cos_vals


# ═════════════════════════════════════════════════════════════════
# 4. FENÊTRAGE GLISSANT + SPLIT TEMPOREL
# ═════════════════════════════════════════════════════════════════

def creer_fenetres_avec_temporel(
    pdl_data: dict,
    horizon: int = 8,
    test_ratio: float = 0.2,
    val_ratio: float = 0.15,
):
    """
    Fenêtre glissante par PDL avec features temporelles et split temporel.

    Split temporel (et non random) : pour chaque PDL, les premières
    fenêtres vont en train, les suivantes en val, les dernières en test.
    Cela évite la fuite d'information : on ne prédit jamais une semaine
    alors qu'on a « vu » la semaine d'après pendant l'entraînement.

    Entrée X : (n_samples, horizon, 9) = 7 conso + sin + cos
    Sortie Y : (n_samples, 7)
    """
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    meta_test, denorm_test = [], []
    X_test_raw, y_test_raw = [], []

    for pdl_id, data in pdl_data.items():
        conso_norm = data["conso_norm"]
        conso_raw = data["conso_raw"]
        dates = data["dates"]
        n_sem = conso_norm.shape[0]

        if n_sem <= horizon:
            continue

        # Features temporelles
        sin_vals, cos_vals = encoder_temporel(dates)

        # Construire les fenêtres
        windows_X, windows_y = [], []
        windows_X_raw, windows_y_raw = [], []
        window_meta = []

        for i in range(n_sem - horizon):
            conso_slice = conso_norm[i: i + horizon]  # (H, 7)
            sin_col = sin_vals[i: i + horizon, None]   # (H, 1)
            cos_col = cos_vals[i: i + horizon, None]   # (H, 1)
            x_row = np.concatenate([conso_slice, sin_col, cos_col], axis=1)  # (H, 9)

            windows_X.append(x_row)
            windows_y.append(conso_norm[i + horizon])
            windows_X_raw.append(conso_raw[i: i + horizon])
            windows_y_raw.append(conso_raw[i + horizon])
            window_meta.append((pdl_id, i + horizon))

        n_windows = len(windows_X)
        if n_windows < 3:
            continue

        # Split temporel chronologique
        n_test = max(1, int(n_windows * test_ratio))
        n_val = max(1, int(n_windows * val_ratio))
        n_train = n_windows - n_test - n_val
        if n_train < 1:
            continue

        for i in range(n_train):
            X_train_list.append(windows_X[i])
            y_train_list.append(windows_y[i])

        for i in range(n_train, n_train + n_val):
            X_val_list.append(windows_X[i])
            y_val_list.append(windows_y[i])

        for i in range(n_train + n_val, n_windows):
            X_test_list.append(windows_X[i])
            y_test_list.append(windows_y[i])
            meta_test.append(window_meta[i])
            denorm_test.append((data["mean"], data["std"]))
            X_test_raw.append(windows_X_raw[i])
            y_test_raw.append(windows_y_raw[i])

    return {
        "X_train": np.array(X_train_list),
        "y_train": np.array(y_train_list),
        "X_val": np.array(X_val_list),
        "y_val": np.array(y_val_list),
        "X_test": np.array(X_test_list),
        "y_test": np.array(y_test_list),
        "meta_test": meta_test,
        "denorm_test": denorm_test,
        "X_test_raw": np.array(X_test_raw) if X_test_raw else np.array([]),
        "y_test_raw": np.array(y_test_raw) if y_test_raw else np.array([]),
    }


# ═════════════════════════════════════════════════════════════════
# 5. DATASET PYTORCH
# ═════════════════════════════════════════════════════════════════

class ConsoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X[:, None, :, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═════════════════════════════════════════════════════════════════
# 6. MODÈLE CNN-LSTM v2
# ═════════════════════════════════════════════════════════════════

INPUT_WIDTH = 9  # 7 jours + sin + cos


class CNNLSTM(nn.Module):
    """
    CNN-LSTM v2.

    Entrée  : (batch, 1, H, 9)  : H semaines × (7 conso + 2 temporel)
    Sortie  : (batch, 7)         : 7 jours prédits (normalisé par PDL)

    Nouveautés v2 :
      - Input 9 colonnes (saisonnalité intégrée)
      - LSTM 2 couches (dépendances plus longues)
      - Tête de prédiction à 2 couches denses
      - Gradient clipping à l'entraînement
    """

    def __init__(self, input_width: int = INPUT_WIDTH):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout_cnn = nn.Dropout(0.15)

        lstm_input = 32 * input_width
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.fc1 = nn.Linear(128, 64)
        self.dropout_fc = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_cnn(x)

        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)


# ═════════════════════════════════════════════════════════════════
# 7. ENTRAÎNEMENT AVEC EARLY STOPPING
# ═════════════════════════════════════════════════════════════════

def entrainer_modele(
    X_train, y_train, X_val, y_val,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 5e-4,
    patience: int = 12,
):
    """
    Entraîne avec early stopping : sauvegarde le meilleur modèle
    (val loss min), s'arrête si pas d'amélioration pendant `patience`
    epochs, et restaure le meilleur état à la fin.
    """
    train_ds = ConsoDataset(X_train, y_train)
    val_ds = ConsoDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CNNLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=6, factor=0.5, min_lr=1e-6,
    )

    historique = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            pred = model(X_b)
            loss = criterion(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b), y_b).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        historique["train_loss"].append(train_loss)
        historique["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    historique["stopped_epoch"] = epoch + 1
    return model, historique


# ═════════════════════════════════════════════════════════════════
# 8. PRÉDICTION & DÉNORMALISATION PAR PDL
# ═════════════════════════════════════════════════════════════════

def predire(model, X_test, denorm_info):
    """Prédit et dénormalise chaque échantillon avec les stats de son PDL."""
    ds = ConsoDataset(X_test, np.zeros((len(X_test), 7)))
    loader = DataLoader(ds, batch_size=64)

    model.eval()
    preds_norm = []
    with torch.no_grad():
        for X_b, _ in loader:
            preds_norm.append(model(X_b).cpu().numpy())
    preds_norm = np.concatenate(preds_norm, axis=0)

    preds_real = np.zeros_like(preds_norm)
    for i, (mean, std) in enumerate(denorm_info):
        preds_real[i] = preds_norm[i] * std + mean

    preds_real = np.maximum(preds_real, 0)
    return preds_real


# ═════════════════════════════════════════════════════════════════
# 9. MÉTRIQUES
# ═════════════════════════════════════════════════════════════════

def calculer_metriques(predictions, cibles):
    """MAE, RMSE, MAPE. Seuil MAPE à 1 kWh pour ne pas gonfler
    l'erreur sur les jours quasi nuls (RS vides)."""
    erreurs = predictions - cibles
    mae = np.abs(erreurs).mean()
    rmse = np.sqrt((erreurs ** 2).mean())
    mask = cibles > 1.0
    mape = np.abs(erreurs[mask] / cibles[mask]).mean() * 100 if mask.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def metriques_par_jour(predictions, cibles):
    jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    resultats = {}
    for j in range(7):
        err = predictions[:, j] - cibles[:, j]
        resultats[jours[j]] = {
            "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
        }
    return resultats


# ═════════════════════════════════════════════════════════════════
# 10. PIPELINE COMPLET
# ═════════════════════════════════════════════════════════════════

def pipeline_complet(
    csv_path: str,
    horizon: int = 8,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 5e-4,
    test_ratio: float = 0.2,
    val_ratio: float = 0.15,
):
    """Pipeline v2 complet."""
    df = charger_donnees(csv_path)
    daily = agreger_journalier(df)
    pdl_data = construire_donnees_pdl(daily, min_semaines=horizon + 3)

    splits = creer_fenetres_avec_temporel(
        pdl_data, horizon=horizon,
        test_ratio=test_ratio, val_ratio=val_ratio,
    )

    model, historique = entrainer_modele(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        epochs=epochs, batch_size=batch_size, lr=lr,
    )

    predictions = predire(model, splits["X_test"], splits["denorm_test"])
    cibles = splits["y_test_raw"]
    metriques = calculer_metriques(predictions, cibles)
    metriques_jours = metriques_par_jour(predictions, cibles)

    return {
        "model": model,
        "historique": historique,
        "predictions": predictions,
        "cibles": cibles,
        "X_test": splits["X_test_raw"],
        "meta_test": splits["meta_test"],
        "metriques": metriques,
        "metriques_jours": metriques_jours,
        "n_pdl": len(pdl_data),
        "n_samples_train": len(splits["X_train"]),
        "n_samples_test": len(splits["X_test"]),
        "n_samples_val": len(splits["X_val"]),
        "horizon": horizon,
        "epochs": historique["stopped_epoch"],
    }