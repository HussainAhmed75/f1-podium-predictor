"""Leak-free, pre-race feature engineering for F1 podium prediction.

Every feature for race R uses only information available *before the race starts*:
- qualifying / starting grid of race R itself (known on Saturday),
- results and standings of races strictly before R.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW = "data/raw/"
CLASSIFIED_RE = r"^(Finished|Lapped|\+\d+ Laps?)$"
START_YEAR = 2006  # qualifying is complete from 2003; 3 seasons of warm-up for rolling stats


def _time_to_sec(s: pd.Series) -> pd.Series:
    """'1:23.456' -> 83.456 ; NaN stays NaN."""
    parts = s.astype("string").str.extract(r"^(?:(\d+):)?(\d+\.\d+)$")
    mins = pd.to_numeric(parts[0], errors="coerce").fillna(0)
    secs = pd.to_numeric(parts[1], errors="coerce")
    return mins * 60 + secs


def load_tables(raw: str = RAW) -> dict[str, pd.DataFrame]:
    names = ["races", "results", "status", "drivers", "constructors", "circuits", "qualifying",
             "driver_standings", "constructor_standings", "pit_stops", "sprint_results"]
    return {n: pd.read_csv(f"{raw}{n}.csv") for n in names}


def base_results(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    races = t["races"][["raceId", "year", "round", "circuitId", "name", "date", "sprint_date"]].copy()
    races["date"] = pd.to_datetime(races["date"])
    df = (t["results"].merge(races, on="raceId").merge(t["status"], on="statusId")
          .sort_values(["date", "raceId", "positionOrder"]).reset_index(drop=True))
    df["dns"] = df["status"].isin(["Did not start", "Withdrew", "Did not qualify", "Did not prequalify"])
    df["classified"] = df["status"].str.match(CLASSIFIED_RE)
    df["dnf"] = ((~df["classified"]) & (~df["dns"])).astype(float)
    df["podium"] = (df["positionOrder"] <= 3).astype(int)
    df["win"] = (df["positionOrder"] == 1).astype(int)
    return df


def _rolling_prev(g: pd.core.groupby.SeriesGroupBy, window: int, how: str = "mean") -> pd.Series:
    """Rolling stat over the previous `window` rows of each group (current row excluded)."""
    return g.transform(lambda s: getattr(s.shift(1).rolling(window, min_periods=1), how)())


def build_features(t: dict[str, pd.DataFrame], start_year: int = START_YEAR) -> pd.DataFrame:
    df = base_results(t)
    df = df[~df["dns"]].copy()

    # ---------- grid & qualifying (known before the race) ----------
    n_cars = df.groupby("raceId")["driverId"].transform("size")
    df["grid_pos"] = np.where(df["grid"] > 0, df["grid"], n_cars)  # 0 = pit-lane start -> back of grid
    df["pitlane_start"] = (df["grid"] == 0).astype(int)
    df["n_cars"] = n_cars

    q = t["qualifying"].copy()
    for c in ["q1", "q2", "q3"]:
        q[c + "_s"] = _time_to_sec(q[c])
    q["best_q"] = q[["q1_s", "q2_s", "q3_s"]].min(axis=1)
    q["pole_q"] = q.groupby("raceId")["best_q"].transform("min")
    q["quali_gap_pct"] = 100 * (q["best_q"] / q["pole_q"] - 1)
    q["reached_q3"] = q["q3_s"].notna().astype(int)
    df = df.merge(q[["raceId", "driverId", "position", "quali_gap_pct", "reached_q3"]]
                  .rename(columns={"position": "quali_pos"}), on=["raceId", "driverId"], how="left")
    df.loc[df["quali_gap_pct"] > 15, "quali_gap_pct"] = np.nan  # in/out-lap artefacts

    # teammate comparison on the grid
    team_best = df.groupby(["raceId", "constructorId"])["grid_pos"].transform("min")
    team_worst = df.groupby(["raceId", "constructorId"])["grid_pos"].transform("max")
    df["grid_vs_teammate"] = np.where(df["grid_pos"] == team_best, df["grid_pos"] - team_worst,
                                      df["grid_pos"] - team_best)
    df["sprint_weekend"] = df["sprint_date"].notna().astype(int)
    df["round_frac"] = df["round"] / df.groupby("year")["round"].transform("max")

    # ---------- driver form (previous races only) ----------
    df = df.sort_values(["date", "raceId"]).reset_index(drop=True)
    gd = df.groupby("driverId")
    df["drv_avg_finish_5"] = _rolling_prev(gd["positionOrder"], 5)
    df["drv_podium_rate_10"] = _rolling_prev(gd["podium"], 10)
    df["drv_win_rate_10"] = _rolling_prev(gd["win"], 10)
    df["drv_dnf_rate_10"] = _rolling_prev(gd["dnf"], 10)
    df["drv_avg_grid_5"] = _rolling_prev(gd["grid_pos"], 5)
    df["drv_gain_5"] = _rolling_prev(gd["grid_pos"], 5) - df["drv_avg_finish_5"]  # avg places gained
    df["drv_experience"] = gd.cumcount()
    df["drv_career_podiums"] = gd["podium"].transform(lambda s: s.shift(1).cumsum()).fillna(0)

    # ---------- constructor form (previous races only) ----------
    team_race = (df.groupby(["raceId", "date", "constructorId"])
                 .agg(team_best=("positionOrder", "min"), team_pod=("podium", "max"),
                      team_dnf=("dnf", "mean"), team_grid=("grid_pos", "min"))
                 .reset_index().sort_values(["date", "raceId"]))
    gt = team_race.groupby("constructorId")
    team_race["team_best_finish_5"] = _rolling_prev(gt["team_best"], 5)
    team_race["team_podium_rate_10"] = _rolling_prev(gt["team_pod"], 10)
    team_race["team_dnf_rate_10"] = _rolling_prev(gt["team_dnf"], 10)
    team_race["team_best_grid_5"] = _rolling_prev(gt["team_grid"], 5)
    df = df.merge(team_race[["raceId", "constructorId", "team_best_finish_5", "team_podium_rate_10",
                             "team_dnf_rate_10", "team_best_grid_5"]], on=["raceId", "constructorId"], how="left")

    # ---------- championship standings after the previous round of the same season ----------
    races = t["races"][["raceId", "year", "round"]]
    nxt = races.copy(); nxt["round"] += 1  # standings after round r are "before round r+1"
    ds = t["driver_standings"].merge(nxt, on="raceId")[["year", "round", "driverId", "position", "points"]]
    ds["leader_pts"] = ds.groupby(["year", "round"])["points"].transform("max")
    ds["drv_pts_share_of_leader"] = np.where(ds["leader_pts"] > 0, ds["points"] / ds["leader_pts"], 0)
    df = df.merge(ds.rename(columns={"position": "drv_champ_pos"})
                  [["year", "round", "driverId", "drv_champ_pos", "drv_pts_share_of_leader"]],
                  on=["year", "round", "driverId"], how="left")
    cs = t["constructor_standings"].merge(nxt, on="raceId")[["year", "round", "constructorId", "position"]]
    df = df.merge(cs.rename(columns={"position": "team_champ_pos"}), on=["year", "round", "constructorId"], how="left")

    # ---------- circuit history (previous visits only) ----------
    df = df.sort_values(["date", "raceId"]).reset_index(drop=True)
    gdc = df.groupby(["driverId", "circuitId"])
    df["drv_circuit_avg_finish"] = gdc["positionOrder"].transform(lambda s: s.shift(1).expanding().mean())
    df["drv_circuit_podiums"] = gdc["podium"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    # how much the grid order is shuffled at this circuit, from earlier races there
    df["abs_moves"] = (df["grid_pos"] - df["positionOrder"]).abs().where(df["classified"])
    circ_race = df.groupby(["raceId", "date", "circuitId"])["abs_moves"].mean().reset_index().sort_values("date")
    circ_race["circuit_shuffle"] = circ_race.groupby("circuitId")["abs_moves"].transform(
        lambda s: s.shift(1).expanding().mean())
    df = df.merge(circ_race[["raceId", "circuit_shuffle"]], on="raceId", how="left")

    df = df[df["year"] >= start_year].reset_index(drop=True)
    return df


FEATURES = [
    "grid_pos", "pitlane_start", "quali_pos", "quali_gap_pct", "reached_q3", "grid_vs_teammate", "n_cars",
    "sprint_weekend", "round_frac",
    "drv_avg_finish_5", "drv_podium_rate_10", "drv_win_rate_10", "drv_dnf_rate_10", "drv_avg_grid_5",
    "drv_gain_5", "drv_experience", "drv_career_podiums",
    "team_best_finish_5", "team_podium_rate_10", "team_dnf_rate_10", "team_best_grid_5",
    "drv_champ_pos", "drv_pts_share_of_leader", "team_champ_pos",
    "drv_circuit_avg_finish", "drv_circuit_podiums", "circuit_shuffle",
]
TARGET = "podium"
