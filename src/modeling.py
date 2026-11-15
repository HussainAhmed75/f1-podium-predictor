"""Model zoo, season-forward cross-validation and race-level evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEED = 42
TEST_YEARS = [2023, 2024, 2025, 2026]
CV_YEARS = [2018, 2019, 2020, 2021, 2022]  # each validated on a model trained on all earlier seasons


def make_models() -> dict:
    return {
        "Grid-only logistic (baseline)": ("grid", Pipeline([("sc", StandardScaler()),
                                                              ("m", LogisticRegression(max_iter=2000))])),
        "Logistic regression": ("all", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                                 ("sc", StandardScaler()),
                                                 ("m", LogisticRegression(C=0.5, max_iter=5000))])),
        "Random forest": ("all", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                           ("m", RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                                        n_jobs=-1, random_state=SEED))])),
        "XGBoost": ("all", XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                         colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=4)),
        "LightGBM": ("all", LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8,
                                           subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)),
    }


def race_top3_hits(frame: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """For each race, pick the 3 highest-scored drivers; return how many actually finished on the podium."""
    f = frame[["raceId", "podium"]].copy()
    f["score"] = score
    f["rank"] = f.groupby("raceId")["score"].rank(ascending=False, method="first")
    return f[f["rank"] <= 3].groupby("raceId")["podium"].sum()


def race_winner_hit(frame: pd.DataFrame, score: np.ndarray) -> float:
    f = frame[["raceId", "win"]].copy()
    f["score"] = score
    top = f.loc[f.groupby("raceId")["score"].idxmax()]
    return float(top["win"].mean())


def metrics(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> dict:
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "podium_hits_per_race": float(race_top3_hits(frame, p).mean()),
            "winner_hit_rate": race_winner_hit(frame, p)}


def season_forward_cv(model, df: pd.DataFrame, cols: list[str], target: str = "podium",
                      years: list[int] = CV_YEARS) -> pd.DataFrame:
    rows = []
    for yr in years:
        tr, va = df[df["year"] < yr], df[df["year"] == yr]
        m = clone(model).fit(tr[cols], tr[target])
        p = m.predict_proba(va[cols])[:, 1]
        rows.append({"year": yr, **metrics(va, va[target].values, p)})
    return pd.DataFrame(rows)
