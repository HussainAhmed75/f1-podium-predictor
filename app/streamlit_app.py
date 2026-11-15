"""F1 Podium Predictor — Streamlit app.

Run from the repo root:   streamlit run app/streamlit_app.py
Needs: results/test_predictions.csv, data/model_table.csv, models/podium_lgbm.joblib (written by the notebook).
"""
import pathlib

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK = "#3987e5", "#d95926", "#f0efec"

st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏁", layout="wide")


@st.cache_data
def load():
    tp = pd.read_csv(ROOT / "results" / "test_predictions.csv")
    table = pd.read_csv(ROOT / "data" / "model_table.csv")
    return tp, table


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "podium_lgbm.joblib")


tp, table = load()
bundle = load_model()
model, FEATURES = bundle["model"], bundle["features"]

st.title("🏁 F1 Podium Predictor")
st.caption("Pre-race podium probabilities from a tuned LightGBM model · data: Jolpica-F1 / Ergast via Kaggle (CC BY-NC-SA 4.0)")

tab1, tab2 = st.tabs(["Race explorer (2023–2026, out-of-sample)", "What-if simulator"])

with tab1:
    c1, c2 = st.columns([1, 3])
    year = c1.selectbox("Season", sorted(tp.year.unique(), reverse=True))
    races = tp[tp.year == year][["round", "name"]].drop_duplicates().sort_values("round")
    race = c2.selectbox("Race", races.name + "  (R" + races["round"].astype(str) + ")", index=len(races) - 1)
    rnd = int(race.split("(R")[1].rstrip(")"))
    d = tp[(tp.year == year) & (tp["round"] == rnd)].sort_values("p_model", ascending=False)

    top3 = d.head(3)
    hits = int(top3.podium.sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Model's top-3 picks on the podium", f"{hits} / 3")
    winner = d[d.positionOrder == 1].iloc[0]
    m2.metric("Winner", f"{winner.forename} {winner.surname}", f"started P{int(winner.grid_pos)}")
    m3.metric("Model's chance for the winner", f"{winner.p_model:.0%}", f"grid-only: {winner.p_grid_baseline:.1%}",
              delta_color="off")

    fig = go.Figure()
    fig.add_bar(x=d.surname, y=d.p_model, name="Model", marker_color=[ORANGE if p else BLUE for p in d.podium],
                text=("P" + d.grid_pos.astype(int).astype(str) + "→P" + d.positionOrder.astype(str)), textposition="outside")
    fig.add_scatter(x=d.surname, y=d.p_grid_baseline, mode="markers", name="Grid-only model",
                    marker=dict(symbol="line-ew-open", size=22, color=INK, line=dict(width=3)))
    fig.update_layout(template="plotly_dark", height=460, yaxis_tickformat=".0%", yaxis_range=[0, 1.1],
                      legend=dict(orientation="h", y=1.1), margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")
    st.caption("Orange = finished on the podium. Label = start → finish. White tick = a model that only knows the grid slot.")
    st.dataframe(d[["surname", "team", "grid_pos", "positionOrder", "p_model", "p_grid_baseline"]]
                 .rename(columns={"surname": "Driver", "team": "Team", "grid_pos": "Grid", "positionOrder": "Finish",
                                  "p_model": "P(podium) model", "p_grid_baseline": "P(podium) grid-only"})
                 .style.format({"P(podium) model": "{:.1%}", "P(podium) grid-only": "{:.1%}"}),
                 width="stretch", hide_index=True)

with tab2:
    st.write("Pick a real driver-race as a starting point, then move the sliders to see how the podium chance reacts.")
    recent = table[table.year == table.year.max()].merge(tp[["raceId", "driverId", "surname"]], on=["raceId", "driverId"], how="left")
    recent = recent.dropna(subset=["surname"])
    recent["key"] = recent.surname + " · R" + recent["round"].astype(str) + " " + recent.name.str.replace(" Grand Prix", " GP")
    key = st.selectbox("Starting point", recent.sort_values(["round", "grid_pos"], ascending=[False, True]).key)
    row = recent[recent.key == key].iloc[0][FEATURES].copy()

    c1, c2, c3 = st.columns(3)
    row["grid_pos"] = c1.slider("Grid position", 1, 22, int(row.grid_pos))
    row["quali_pos"] = row["grid_pos"]
    gap = 0.0 if pd.isna(row.quali_gap_pct) else float(row.quali_gap_pct)
    row["quali_gap_pct"] = c1.slider("Qualifying gap to pole (%)", 0.0, 4.0, min(gap, 4.0), 0.05)
    row["team_podium_rate_10"] = c2.slider("Team podium rate, last 10 races", 0.0, 1.0,
                                           float(0 if pd.isna(row.team_podium_rate_10) else row.team_podium_rate_10), 0.1)
    row["team_best_finish_5"] = c2.slider("Team best finish, avg of last 5", 1.0, 20.0,
                                          float(10 if pd.isna(row.team_best_finish_5) else row.team_best_finish_5), 0.5)
    row["drv_podium_rate_10"] = c3.slider("Driver podium rate, last 10 races", 0.0, 1.0,
                                          float(0 if pd.isna(row.drv_podium_rate_10) else row.drv_podium_rate_10), 0.1)
    row["drv_pts_share_of_leader"] = c3.slider("Driver points ÷ leader's points", 0.0, 1.0,
                                               float(0 if pd.isna(row.drv_pts_share_of_leader) else row.drv_pts_share_of_leader), 0.05)
    p = model.predict_proba(pd.DataFrame([row.values], columns=FEATURES).astype(float))[:, 1][0]
    st.metric("Predicted podium probability", f"{p:.0%}")
    st.progress(float(p))
    st.caption("The what-if model is the same tuned LightGBM retrained on all seasons 2006–2026.")
