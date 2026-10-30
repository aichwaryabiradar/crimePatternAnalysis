"""Page 1 - Crime-pattern statistics for the whole city."""

import streamlit as st

from views import charts

W = "stretch"


def show(df, info, P):
    st.subheader("Crime patterns in Chicago")
    st.caption(f"{info['rows_clean']:,} incidents · {info['n_days']:,} days "
               f"({info['date_min']} to {info['date_max']}) · {info['n_areas']} community areas")

    st.markdown("##### 1. Frequency analysis")
    st.caption("Relative frequency estimates probability: P(A) = n(A) / N")
    c1, c2 = st.columns(2)
    f = P["category_freq"]
    c1.plotly_chart(charts.bars(f.index, f["probability"], "Crime categories"), width=W)
    a = P["area_freq"]
    c2.plotly_chart(charts.bars(a.index, a["probability"], "Top 10 community areas",
                                horizontal=True), width=W)
    c1, c2 = st.columns(2)
    s = P["slot_freq"]
    c1.plotly_chart(charts.bars(s.index, s["probability"], "Incidents by time slot"), width=W)
    d = P["day_freq"]
    c2.plotly_chart(charts.bars(d.index, d["probability"], "Incidents by weekday"), width=W)

    st.markdown("##### 2. Conditional probability")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.heatmap(P["P(category|time_slot)"],
                                   "P(category | time slot): each row sums to 1"), width=W)
    with c2:
        b = P["bayes"]
        st.markdown("**Bayes' theorem example**")
        st.latex(r"P(V\mid D)=\frac{P(D\mid V)\,P(V)}{P(D)}=\frac{%.3f\times%.3f}{%.3f}=%.3f"
                 % (b["P(DOMESTIC|VIOLENT)"], b["P(VIOLENT)"], b["P(DOMESTIC)"],
                    b["P(VIOLENT|DOMESTIC) Bayes"]))
        st.caption(f"V = violent, D = domestic. Knowing an incident is domestic raises "
                   f"P(violent) from {b['P(VIOLENT)']:.1%} to {b['P(VIOLENT|DOMESTIC) count']:.1%}.")
        st.dataframe(P["P(arrest|category)"].rename("P(arrest | category)").to_frame().round(3),
                     width=W)

    st.markdown("##### 3. Mean and variance")
    dd = P["daily"]
    c = st.columns(4)
    c[0].metric("Mean incidents per day", f"{dd['mean']:.2f}")
    c[1].metric("Variance", f"{dd['variance']:.2f}")
    c[2].metric("Standard deviation", f"{dd['std']:.2f}")
    c[3].metric("Coefficient of variation", f"{dd['cv']:.3f}")
    c1, c2 = st.columns(2)
    c1.markdown("**Daily incidents by weekday**")
    c1.dataframe(P["daily_by_weekday"].round(2), width=W)
    c2.markdown("**Incidents per place-time cell, by time slot**")
    c2.dataframe(P["cells_by_slot"].round(2), width=W)
    st.caption("Afternoon and evening cells have the highest mean and also the largest variance: "
               "a few places carry many more incidents than others. These are the hotspots the "
               "models learn to predict.")

    st.markdown("##### 4. Correlation")
    c1, c2 = st.columns([2, 3])
    t = P["correlation"].copy()
    t["significant"] = t["p_value"] < 0.05
    c1.markdown("**Correlation of each feature with the cell's incident count**")
    c1.dataframe(t.style.format({"r": "{:.3f}", "p_value": "{:.2e}"}), width=W, hide_index=True)
    c1.caption("H₀: ρ = 0 (Pearson test). Time slot has the strongest linear link; location "
               "matters, but not linearly, which is why tree models do better.")
    c2.plotly_chart(charts.heatmap(P["corr_matrix"], "Correlation matrix", colorscale="RdBu_r",
                                   zmin=-1, zmax=1, height=360), width=W)
