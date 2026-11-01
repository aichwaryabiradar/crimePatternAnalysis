"""Page 3 - Hotspot predictor: the user chooses a place (and a time) and gets the
crime-pattern analysis plus the predictions of all three models."""

import pandas as pd
import streamlit as st

from src import hotspot
from src.config import COMMUNITY_AREAS, DAYS, TIME_SLOTS
from src.models import MODEL_NAMES
from views import charts

W = "stretch"


def show(df, cells, models, threshold, R):
    st.subheader("Hotspot predictor")
    areas = sorted(COMMUNITY_AREAS.values())
    area = st.selectbox("Choose a place (Chicago community area; type to search)", areas,
                        index=areas.index("Logan Square"))

    # ---------- 1. crime-pattern analysis for the place ----------
    prof = hotspot.area_profile(df, cells, area)
    st.markdown(f"##### 1. Crime pattern in {area}")
    c = st.columns(4)
    c[0].metric("Incidents in sample", f"{prof['n']:,}",
                help=f"{prof['share']:.1%} of the city total")
    c[1].metric("Rank among 77 areas", f"#{prof['rank']}")
    c[2].metric("Mean incidents / day (variance)",
                f"{prof['daily']['mean']:.2f} ({prof['daily']['variance']:.2f})")
    c[3].metric("P(arrest)", f"{prof['arrest_rate']:.1%}",
                delta=f"{(prof['arrest_rate'] - prof['city_arrest_rate']) * 100:+.1f} pp vs city",
                delta_color="off")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.grouped_bars(prof["category"], f"P(category | {area}) vs city",
                                        y="Probability"), width=W)
    c2.plotly_chart(charts.heatmap(prof["grid"], "Observed incidents by weekday and time slot",
                                   fmt=".0f", colorscale="Oranges"), width=W)

    # ---------- 2. model predictions for the place ----------
    st.markdown(f"##### 2. Predicted high-risk periods in {area}")
    grids, actual = hotspot.area_risk_grid(models, cells, area, threshold)
    cols = st.columns(3)
    for col, name in zip(cols, MODEL_NAMES):
        acc = ((grids[name].values >= 0.5) == actual.values).mean()
        col.plotly_chart(charts.heatmap(grids[name], f"{name} · P(high risk)", zmin=0, zmax=1,
                                        colorscale="YlOrRd", height=280, showscale=False),
                         width=W)
        col.caption(f"Agrees with the observed pattern in {acc:.0%} of the 28 cells · "
                    f"overall test accuracy {R['metrics'][name]['accuracy']:.1%}")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Riskiest periods (Random Forest)**")
        st.dataframe(hotspot.top_periods(grids).style.format({"P(high risk)": "{:.2f}"}),
                     width=W, hide_index=True)
    with c2:
        st.plotly_chart(charts.heatmap(actual, "Observed: high-risk cells (1 = high risk)",
                                       fmt=".0f", colorscale=[[0, "#f2f2f0"], [1, "#eb6834"]],
                                       zmin=0, zmax=1, height=280, showscale=False), width=W)

    # ---------- 3. a specific time ----------
    st.markdown("##### 3. Check a specific time")
    c1, c2 = st.columns(2)
    day = c1.selectbox("Day", DAYS, index=4)
    slot = c2.selectbox("Time slot", TIME_SLOTS, index=3)
    rows = []
    for name in MODEL_NAMES:
        p = grids[name].loc[slot, day]
        rows.append({"Model": name, "P(high risk)": p, "Prediction": "HIGH RISK" if p >= 0.5 else "low risk"})
    obs = int(cells[(cells["area"] == area) & (cells["day"] == day) &
                    (cells["time_slot"] == slot)]["incidents"].iloc[0])
    st.dataframe(pd.DataFrame(rows).style.format({"P(high risk)": "{:.2f}"}), width=W,
                 hide_index=True)
    st.caption(f"Observed: {obs} incidents in this cell → "
               f"{'high risk' if obs >= threshold else 'low risk'} (threshold {threshold:.0f}).")

    # ---------- 4. riskiest locations at that time ----------
    st.markdown(f"##### 4. High-risk locations on {day}, {slot}")
    top = hotspot.riskiest_areas(models, cells, DAYS.index(day), TIME_SLOTS.index(slot))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.area_map(top, "Random Forest", "Predicted risk by community area "
                                    f"(Random Forest; ★ = {area})", highlight=area), width=W)
    short = {"Logistic Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF"}
    t = top[["area", "incidents"] + MODEL_NAMES].head(10).rename(
        columns={"area": "Area", "incidents": "Observed", **short})
    c2.markdown("**Top 10 predicted hotspots** (P(high risk) by model)")
    c2.dataframe(t.style.format({v: "{:.2f}" for v in short.values()}), width=W, hide_index=True)
    c2.caption("LR = Logistic Regression, DT = Decision Tree, RF = Random Forest. "
               f"Observed = incidents in this cell; ≥ {threshold:.0f} means high risk.")
