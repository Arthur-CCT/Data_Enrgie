# --- RP/RS clustering pipeline (COPY/PASTE) ---

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ====== CONFIG ======
DATA_PATH = "datas/courbes-de-charges-fictives-res2-6-9.csv"
COL_PDL = "pdl_id"
COL_DT  = "datetime"
COL_PWR = "p_kw"

TZ = "Europe/Paris"
STEP_HOURS = 0.5  # 30 min

# ====== HELPERS ======
def parse_local_datetime(series: pd.Series, tz_name: str) -> pd.Series:
    """
    Parse datetimes robustly on unique values, then broadcast.
    - with timezone suffix: convert to tz_name, then drop tz (local naive)
    - without timezone suffix: keep as local naive
    """
    s = series.astype("string").str.strip()
    unique_vals = pd.Index(s.dropna().unique())

    parsed_unique = pd.Series(pd.NaT, index=unique_vals, dtype="datetime64[ns]")
    has_tz = unique_vals.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False)

    if has_tz.any():
        aware = pd.to_datetime(unique_vals[has_tz], errors="coerce", utc=True)
        parsed_unique.loc[unique_vals[has_tz]] = aware.tz_convert(tz_name).tz_localize(None).to_numpy()

    if (~has_tz).any():
        naive = pd.to_datetime(unique_vals[~has_tz], errors="coerce")
        parsed_unique.loc[unique_vals[~has_tz]] = naive.to_numpy()

    return pd.to_datetime(s.map(parsed_unique), errors="coerce")

# ====== LOAD (faster + robust to "ID"/"id") ======
# Read all, then normalize headers (handles quotes, spaces, case)
raw = pd.read_csv(DATA_PATH, sep=",")
raw.columns = raw.columns.str.strip().str.replace('"', "", regex=False).str.lower()

raw = raw.rename(columns={"id": COL_PDL, "horodate": COL_DT, "valeur": COL_PWR})

missing = [c for c in [COL_PDL, COL_DT, COL_PWR] if c not in raw.columns]
if missing:
    raise ValueError(f"Missing columns after rename: {missing}. Available: {raw.columns.tolist()}")

# Parse datetime and convert to Europe/Paris LOCAL TIME naive
# (robust to mixed timezone offsets and faster via unique-value parsing)
raw[COL_DT] = parse_local_datetime(raw[COL_DT], TZ)

df = raw.dropna(subset=[COL_PDL, COL_DT, COL_PWR]).copy()
df[COL_PWR] = pd.to_numeric(df[COL_PWR], errors="coerce")
df = df.dropna(subset=[COL_PWR])

# ====== TIME FEATURES (faster: use datetime64 not python date objects) ======
df["date"] = df[COL_DT].dt.floor("D")  # datetime64[ns], faster than .dt.date
df["dow"] = df[COL_DT].dt.dayofweek
df["is_weekend"] = df["dow"] >= 5
df["hh_index"] = ((df[COL_DT].dt.hour * 60) + df[COL_DT].dt.minute) // 30  # 0..47

print("n_rows:", len(df))
print("n_clients:", df[COL_PDL].nunique())

# ====== DAILY AGG (kW -> kWh/day) ======
daily = (
    df.assign(energy_kwh_step=df[COL_PWR] * STEP_HOURS)
      .groupby([COL_PDL, "date"], as_index=False, sort=False)
      .agg(
          daily_kwh=("energy_kwh_step", "sum"),
          daily_mean_kw=(COL_PWR, "mean"),
          daily_max_kw=(COL_PWR, "max"),
          n_steps=(COL_PWR, "size"),
      )
)

# ====== ACTIVE DAY THRESHOLD PER PDL ======
q20_by_pdl = daily[daily["daily_kwh"] > 0].groupby(COL_PDL, sort=False)["daily_kwh"].quantile(0.2)
daily["th_pdl"] = daily[COL_PDL].map(q20_by_pdl)
daily["is_active_day"] = (daily["daily_kwh"] >= daily["th_pdl"]).fillna(False)

# ====== FEATURES: ACTIVITY ======
activity = (
    daily.groupby(COL_PDL, as_index=False, sort=False)
         .agg(
             n_days=("date", "size"),
             n_active_days=("is_active_day", "sum"),
             active_day_rate=("is_active_day", "mean"),
             mean_daily_kwh=("daily_kwh", "mean"),
             p95_daily_kwh=("daily_kwh", lambda s: s.quantile(0.95)),
             cv_daily_kwh=("daily_kwh", lambda s: (s.std() / s.mean()) if s.mean() != 0 else np.nan),
         )
)

# ====== FEATURES: RUNS / GAPS ======
def runs_and_gaps(active_series: pd.Series):
    runs, gaps = [], []
    run = gap = 0
    for v in active_series.astype(bool):
        if v:
            run += 1
            if gap > 0:
                gaps.append(gap); gap = 0
        else:
            gap += 1
            if run > 0:
                runs.append(run); run = 0
    if run > 0: runs.append(run)
    if gap > 0: gaps.append(gap)
    return pd.Series({
        "n_runs": len(runs),
        "mean_run_len": float(np.mean(runs)) if runs else 0.0,
        "max_run_len": float(np.max(runs)) if runs else 0.0,
        "mean_gap_len": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap_len": float(np.max(gaps)) if gaps else 0.0,
    })

runs_stats = (
    daily.sort_values([COL_PDL, "date"])
         .groupby(COL_PDL, sort=False)["is_active_day"]
         .apply(runs_and_gaps)
         .unstack()
         .reset_index()
)

# ====== FEATURES: WEEKDAY vs WEEKEND ======
# daily["date"] is already datetime64; no need for pd.to_datetime
daily["dow"] = daily["date"].dt.dayofweek
daily["is_weekend"] = daily["dow"] >= 5

week_pattern = (
    daily.groupby([COL_PDL, "is_weekend"], as_index=False, sort=False)
         .agg(active_rate=("is_active_day", "mean"),
              mean_kwh=("daily_kwh", "mean"))
         .pivot(index=COL_PDL, columns="is_weekend")
)

week_pattern.columns = [f"{a}_{'weekend' if b else 'weekday'}" for a, b in week_pattern.columns]
week_pattern = week_pattern.reset_index()

# ====== FEATURES: SEASONALITY RATIOS ======
daily2 = daily.copy()
daily2["month"] = daily2["date"].dt.month

def season_from_month(m):
    if m in (12, 1, 2): return "winter"
    if m in (6, 7, 8):  return "summer"
    return "mid"

daily2["season"] = daily2["month"].map(season_from_month)

season_stats = (
    daily2.groupby([COL_PDL, "season"], as_index=False, sort=False)
          .agg(mean_daily_kwh=("daily_kwh", "mean"))
          .pivot(index=COL_PDL, columns="season", values="mean_daily_kwh")
          .reset_index()
)

for c in ["winter", "summer", "mid"]:
    if c not in season_stats.columns:
        season_stats[c] = 0.0

global_mean = daily2.groupby(COL_PDL, as_index=False, sort=False).agg(mean_daily_kwh_global=("daily_kwh", "mean"))
season_stats = season_stats.merge(global_mean, on=COL_PDL, how="left", validate="one_to_one")

eps = 1e-9
season_stats["r_mid"]    = season_stats["mid"]    / (season_stats["mean_daily_kwh_global"] + eps)
season_stats["r_summer"] = season_stats["summer"] / (season_stats["mean_daily_kwh_global"] + eps)
season_stats["r_winter"] = season_stats["winter"] / (season_stats["mean_daily_kwh_global"] + eps)
season_stats = season_stats[[COL_PDL, "r_mid", "r_summer", "r_winter"]]

# ====== MERGE ALL ======
features_pdl = (
    activity
    .merge(runs_stats, on=COL_PDL, how="left", validate="one_to_one")
    .merge(week_pattern, on=COL_PDL, how="left", validate="one_to_one")
    .merge(season_stats, on=COL_PDL, how="left", validate="one_to_one")
)

assert features_pdl[COL_PDL].is_unique, "Merge exploded: more than one row per PDL"

features_pdl["seasonality_amp"] = (
    features_pdl[["r_mid", "r_summer", "r_winter"]].max(axis=1)
    - features_pdl[["r_mid", "r_summer", "r_winter"]].min(axis=1)
)
features_pdl["winter_minus_summer"] = features_pdl["r_winter"] - features_pdl["r_summer"]

# ====== CLUSTERING ======
feature_cols = [
    "active_day_rate",
    "n_runs", "mean_run_len", "max_run_len",
    "mean_gap_len", "max_gap_len",
    "mean_daily_kwh", "p95_daily_kwh", "cv_daily_kwh",
    "active_rate_weekday", "active_rate_weekend",
    "mean_kwh_weekday", "mean_kwh_weekend",
    "winter_minus_summer", "seasonality_amp",
    "r_mid", "r_summer", "r_winter",
]

X = features_pdl[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_scaled = StandardScaler().fit_transform(X)

scores = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

best_k_sil = max(scores, key=scores.get)
print("silhouette scores:", scores)
print("best k (silhouette):", best_k_sil)

K = 10
kmeans = KMeans(n_clusters=K, n_init=50, random_state=42)
features_pdl["cluster"] = kmeans.fit_predict(X_scaled)

# ====== REPORTS ======
summary = (
    features_pdl.groupby("cluster")
                .agg(
                    n_clients=(COL_PDL, "size"),
                    active_day_rate=("active_day_rate", "mean"),
                    max_gap_len=("max_gap_len", "mean"),
                    winter_minus_summer=("winter_minus_summer", "mean"),
                    cv_daily_kwh=("cv_daily_kwh", "mean"),
                    r_summer=("r_summer", "mean"),
                )
                .sort_index()
)

print("\nsummary:\n", summary)

# ====== EXPORT LABELS (edit mapping after interpreting clusters) ======
cluster_to_label = {
    0: 0, 3: 0, 5: 0, 8: 0, 9: 0,   # RP
    1: 1, 2: 1, 4: 1, 6: 1, 7: 1,   # RS / atypique
}

labels_df = features_pdl[[COL_PDL, "cluster"]].copy()
labels_df["label"] = labels_df["cluster"].map(cluster_to_label)

output_df = labels_df.rename(columns={COL_PDL: "id"})[["id", "label", "cluster"]]
output_path = "outputs/final_clustering_labels.csv"
output_df.to_csv(output_path, index=False, sep=",")

print("\nSaved:", output_path)
print(output_df.head())
