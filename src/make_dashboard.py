"""Interactive HTML dashboard (Plotly, works offline) -> reports/f1_dashboard.html
Run from the repo root after the notebook:  python src/make_dashboard.py
"""
import json
import pathlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from features import load_tables, base_results  # noqa: E402
from style import SURFACE, INK, INK_2, MUTED, GRID, BLUE, ORANGE, AQUA, SERIES  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
t = load_tables(str(ROOT / "data" / "raw") + "/")
res = base_results(t)
M = json.loads((ROOT / "results" / "metrics.json").read_text())
tp = pd.read_csv(ROOT / "results" / "test_predictions.csv")


def layout(fig, title, sub, h=430):
    fig.update_layout(template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=h,
                      font=dict(family="Inter, Segoe UI, Arial", color=INK, size=13),
                      title=dict(text=f"<b>{title}</b><br><span style='font-size:13px;color:{INK_2}'>{sub}</span>",
                                 x=0.01, xanchor="left"),
                      margin=dict(l=60, r=30, t=90, b=50), hoverlabel=dict(bgcolor="#262624"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False); fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


figs = []

# 1 podium chance by grid
mod = res[(res.year >= 2010) & ~res.dns & (res.grid > 0)]
pg = mod.groupby("grid").agg(p=("podium", "mean"), n=("podium", "size")).loc[1:20].reset_index()
f = go.Figure(go.Bar(x=pg.grid, y=pg.p, marker_color=[ORANGE if g <= 3 else BLUE for g in pg.grid],
                     customdata=pg.n, hovertemplate="Grid P%{x}<br>Podium chance %{y:.1%}<br>%{customdata} starts<extra></extra>"))
f.update_yaxes(tickformat=".0%"); f.update_xaxes(title="Starting grid position", dtick=1)
figs.append(layout(f, "Where you start ≈ where you finish", "Chance of a podium by grid slot, 2010–2026"))

# 2 DNF rate per season
dnf = res[~res.dns].groupby("year").dnf.mean().reset_index()
f = go.Figure(go.Scatter(x=dnf.year, y=dnf.dnf, mode="lines", line=dict(color=BLUE, width=2.5),
                         hovertemplate="%{x}: %{y:.1%} DNF<extra></extra>"))
f.add_trace(go.Scatter(x=[2026], y=[dnf.dnf.iloc[-1]], mode="markers", marker=dict(color=ORANGE, size=12),
                       hovertemplate="2026: %{y:.1%}<extra></extra>", showlegend=False))
f.update_yaxes(tickformat=".0%"); f.update_layout(showlegend=False)
figs.append(layout(f, "Cars got bulletproof — until the 2026 regulations", "Share of starters that did not finish, per season"))

# 3 2026 title fight (all drivers, click legend to toggle)
ds = t["driver_standings"].merge(t["races"][["raceId", "year", "round", "name"]], on="raceId")
s26 = ds[ds.year == 2026].merge(t["drivers"][["driverId", "surname"]], on="driverId")
order = s26[s26["round"] == s26["round"].max()].sort_values("position").surname.tolist()
f = go.Figure()
for i, n in enumerate(order):
    d = s26[s26.surname == n].sort_values("round")
    f.add_trace(go.Scatter(x=d["round"], y=d.points, name=n, mode="lines+markers",
                           line=dict(width=2.5 if i < 5 else 1.2, color=SERIES[i] if i < 5 else "#55554f"),
                           marker=dict(size=5), visible=True if i < 8 else "legendonly", customdata=d.name,
                           hovertemplate=f"<b>{n}</b><br>%{{customdata}}<br>%{{y:.0f}} pts<extra></extra>"))
f.update_xaxes(title="Round"); f.update_yaxes(title="Points")
figs.append(layout(f, "2026 drivers' championship", "Points after each round — click names in the legend to add/remove drivers", 520))

# 4 race explorer: model vs grid-only probabilities for every test race
tp["label"] = tp.year.astype(str) + " · R" + tp["round"].astype(str).str.zfill(2) + " · " + tp.name.str.replace(" Grand Prix", " GP")
races = tp[["raceId", "label"]].drop_duplicates().sort_values("raceId")
f = go.Figure()
for k, (rid, lab) in enumerate(races.itertuples(index=False)):
    d = tp[tp.raceId == rid].sort_values("p_model", ascending=False).head(12)
    txt = "P" + d.grid_pos.astype(int).astype(str) + " → P" + d.positionOrder.astype(str)
    vis = k == len(races) - 1
    f.add_trace(go.Bar(x=d.surname, y=d.p_model, name="Model", marker_color=[ORANGE if p else BLUE for p in d.podium],
                       text=txt, textposition="outside", visible=vis, customdata=d.p_grid_baseline,
                       hovertemplate="<b>%{x}</b><br>model %{y:.1%}<br>grid-only %{customdata:.1%}<br>%{text}<extra></extra>"))
    f.add_trace(go.Scatter(x=d.surname, y=d.p_grid_baseline, name="Grid-only model", mode="markers",
                           marker=dict(symbol="line-ew-open", size=26, color=INK, line=dict(width=3)), visible=vis,
                           hoverinfo="skip"))
buttons = [dict(label=lab, method="update",
                args=[{"visible": [j // 2 == k for j in range(2 * len(races))]}])
           for k, lab in enumerate(races.label)]
f.update_layout(updatemenus=[dict(buttons=buttons[::-1], direction="down", x=1, xanchor="right", y=1.18,
                                  bgcolor="#262624", font=dict(color=INK))],
                legend=dict(orientation="h", y=-0.2))
f.update_yaxes(tickformat=".0%", range=[0, 1.08])
figs.append(layout(f, "Race explorer — 2023 to 2026 (out-of-sample)",
                   "Bars = model's pre-race podium chance (orange = finished on the podium) · white tick = grid-only model · label = start → finish", 540))

# 5 model scoreboard
tm, tb = M["test"]["model"], M["test"]["grid_only_baseline"]
rows = [("ROC-AUC ↑", tm["roc_auc"], tb["roc_auc"], ".3f"), ("PR-AUC ↑", tm["pr_auc"], tb["pr_auc"], ".3f"),
        ("Log-loss ↓", tm["log_loss"], tb["log_loss"], ".3f"), ("Brier ↓", tm["brier"], tb["brier"], ".4f"),
        ("Winner picked ↑", tm["winner_hit_rate"], tb["winner_hit_rate"], ".1%"),
        ("Podium finishers in top-3 picks ↑", tm["podium_hits_per_race"], tb["podium_hits_per_race"], ".2f")]
f = go.Figure(go.Table(header=dict(values=["<b>Test 2023–2026</b>", "<b>Tuned LightGBM</b>", "<b>Grid-only model</b>"],
                                   fill_color="#262624", font=dict(color=INK), align="left", height=34),
                       cells=dict(values=[[r[0] for r in rows], [format(r[1], r[3]) for r in rows],
                                          [format(r[2], r[3]) for r in rows]],
                                  fill_color=SURFACE, font=dict(color=[INK_2, ORANGE, INK_2], size=14), align="left", height=32)))
figs.append(layout(f, "Model scoreboard", f"{M['facts']['test_races']} held-out races; nothing from these seasons was used for training or tuning", 360))

from plotly.offline import get_plotlyjs
html = [f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>F1 Podium Predictor — Dashboard</title>
<script>{{PLOTLYJS}}</script>
<style>body{{background:{SURFACE};color:{INK};font-family:Inter,Segoe UI,Arial;margin:0;padding:32px 48px;max-width:1200px;margin:auto}}
h1{{margin:0 0 4px;font-size:30px}} p.lead{{color:{INK_2};margin:0 0 24px}} .card{{border:1px solid {GRID};border-radius:14px;margin:18px 0;padding:6px}}
footer{{color:{MUTED};font-size:12px;margin:30px 0}}</style></head><body>
<h1>🏁 F1 Podium Predictor</h1><p class='lead'>77 seasons of Formula 1 (1950 → Brazil 2026) · analysis + a leak-free pre-race podium model · Hussain Ahmed</p>"""]
for i, fg in enumerate(figs):
    html.append("<div class='card'>" + fg.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig{i}",
                                                   config={"displaylogo": False}) + "</div>")
html.append("<footer>Data: Jolpica-F1 / Ergast via Kaggle (Samuel_Pbq), CC BY-NC-SA 4.0. Non-commercial portfolio project.</footer></body></html>")
CDN = "--cdn" in sys.argv  # light version (loads Plotly from the internet) for GitHub
out = ROOT / "reports" / "f1_dashboard.html"
out.parent.mkdir(exist_ok=True)
page = "\n".join(html)
if CDN:
    page = page.replace("<script>{PLOTLYJS}</script>", "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>")
else:
    page = page.replace("{PLOTLYJS}", get_plotlyjs())
out.write_text(page, encoding="utf-8")
print("saved", out)
