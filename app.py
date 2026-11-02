"""
Crime Hotspot Analysis: probability, statistics and ML model comparison.
Run:  streamlit run app.py
"""

import streamlit as st

from src import data, evaluation, models, statistics
from views import comparison, patterns, predictor

st.set_page_config(page_title="Crime Hotspot Analysis", page_icon="📍", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;max-width:1300px}"
            "[data-testid='stMetricValue']{font-size:1.4rem}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data ...")
def load():
    df, info = data.load_incidents()
    cells = data.build_cells(df)
    pats = statistics.city_patterns(df, data.daily_counts(df), cells)
    return df, info, cells, pats


@st.cache_resource(show_spinner="Training the three models ...")
def train(_cells):
    tr, te, thr = models.split_and_label(_cells)
    ms = models.train_models(tr)
    res = evaluation.compare_models(te, models.predict_proba(ms, te))
    return tr, te, thr, ms, res


df, info, cells, P = load()
tr, te, thr, ms, R = train(cells)

st.sidebar.title("📍 Crime Hotspot Analysis")
page = st.sidebar.radio("Page", ["Hotspot predictor", "Crime patterns", "Model comparison"])
st.sidebar.caption("Chicago Police Department incidents, 2017–2019.")

if page == "Hotspot predictor":
    predictor.show(df, cells, ms, thr, R)
elif page == "Crime patterns":
    patterns.show(df, info, P)
else:
    comparison.show(tr, te, thr, R)
