"""Step 3-4 - Sampling and the three models.

Task: is a (community area, weekday, time slot) cell a HIGH-RISK hotspot?
      high risk = incident count in the top 25 % of training cells.
Features: area location (latitude, longitude), weekday, time slot, weekend flag."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import RISK_QUANTILE, SEED, TEST_SIZE

FEATURES = ["latitude", "longitude", "day_of_week", "slot", "is_weekend"]
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]
MODEL_INFO = {
    "Logistic Regression": "Baseline. Linear model of the log-odds of high risk.",
    "Decision Tree": "One tree of if/else rules (max depth 6).",
    "Random Forest": "300 trees on bootstrap samples; the vote share is the probability.",
}


def build_model(name, seed=SEED):
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                      random_state=seed)
    raise ValueError(name)


def split_and_label(cells, test_size=TEST_SIZE, seed=SEED):
    """Random 70/30 split of the cells. The high-risk threshold is the 75th percentile
    of TRAINING counts only, then the split is checked for class balance."""
    train, test = train_test_split(cells, test_size=test_size, random_state=seed)
    threshold = float(train["incidents"].quantile(RISK_QUANTILE))
    train, test = train.copy(), test.copy()
    for part in (train, test):
        part["high_risk"] = (part["incidents"] >= threshold).astype(int)
    return train, test, threshold


def train_models(train, seed=SEED):
    return {name: build_model(name, seed).fit(train[FEATURES], train["high_risk"])
            for name in MODEL_NAMES}


def predict_proba(models, X):
    """P(high risk) from every model for the rows of X."""
    return {name: m.predict_proba(X[FEATURES])[:, 1] for name, m in models.items()}
