"""Page 2 - Do the models really differ, or is it test-sample variation?"""

import pandas as pd
import streamlit as st

from src.models import MODEL_INFO
from views import charts

W = "stretch"


def show(train, test, threshold, R):
    st.subheader("Model comparison")
    st.markdown("**Task:** predict whether a place-time cell (community area × weekday × time "
                f"slot) is a **high-risk hotspot**, meaning its incident count is ≥ {threshold:.0f} "
                "(the top 25% of training cells).")
    for name, txt in MODEL_INFO.items():
        st.markdown(f"- **{name}**: {txt}")

    st.markdown("##### 1. Sampling")
    c = st.columns(4)
    c[0].metric("Training cells", f"{len(train):,}")
    c[1].metric("Held-out test cells", f"{len(test):,}")
    c[2].metric("High-risk share (train)", f"{train['high_risk'].mean():.1%}")
    c[3].metric("High-risk share (test)", f"{test['high_risk'].mean():.1%}")
    st.caption("Random 70/30 split with a fixed seed. All three models are trained on the same "
               "training cells and scored on the same test cells. A model that always says "
               f"'not high risk' would score {1 - test['high_risk'].mean():.1%}.")

    st.markdown("##### 2. Estimation: accuracy with 95% confidence intervals")
    t = R["table"].copy()
    st.dataframe(t.style.format({c: "{:.3f}" for c in t.columns if c != "Model"}),
                 width=W, hide_index=True)
    st.plotly_chart(charts.accuracy_ci(t, "Test accuracy and 95% Wilson CI"), width=W)
    cols = st.columns(3)
    for col, (name, m) in zip(cols, R["metrics"].items()):
        cm = pd.DataFrame(m["confusion"], index=["actual low", "actual high"],
                          columns=["pred low", "pred high"])
        col.plotly_chart(charts.heatmap(cm, name, fmt=".0f", height=230, showscale=False),
                         width=W)

    st.markdown("##### 3. Hypothesis test: McNemar's test for each pair")
    st.markdown("H₀: the two models have the same accuracy. Because all models predict the "
                "**same** test cells, the test is paired: only cells where exactly one model is "
                "right count. Under H₀, c ~ Binomial(b + c, ½). With 3 comparisons, each is "
                f"tested at α/3 = {R['alpha_adjusted']:.4f} (Bonferroni).")
    p = R["pairs"].copy()
    st.dataframe(p.style.format({"Accuracy difference": "{:+.3f}", "CI low": "{:+.3f}",
                                 "CI high": "{:+.3f}", "McNemar p": "{:.2e}"}),
                 width=W, hide_index=True)

    st.markdown("##### 4. Bootstrap 95% CI of the accuracy difference")
    pair = st.selectbox("Comparison", list(R["bootstrap"]), index=2)
    row = p[p["Comparison"] == pair].iloc[0]
    st.plotly_chart(charts.bootstrap_hist(R["bootstrap"][pair], row["CI low"], row["CI high"],
                                          f"{pair}: 2,000 bootstrap resamples of the test set"),
                    width=W)

    st.markdown("##### Conclusion")
    for _, r in p.iterrows():
        if r["Significant (α/3)"]:
            st.success(f"**{r['Comparison']}**: {r['Accuracy difference']:+.1%} "
                       f"(95% CI {r['CI low']:+.1%} to {r['CI high']:+.1%}), McNemar p = "
                       f"{r['McNemar p']:.1e}. A real difference, not test-sample variation.")
        else:
            st.warning(f"**{r['Comparison']}**: {r['Accuracy difference']:+.1%} "
                       f"(95% CI {r['CI low']:+.1%} to {r['CI high']:+.1%}), McNemar p = "
                       f"{r['McNemar p']:.2f}. The difference **could be due to test-sample "
                       "variation**; we cannot claim one is better.")
