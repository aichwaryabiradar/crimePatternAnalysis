# Crime Hotspot Analysis

Chicago Police Department crime incidents (2017–2019) are used to:

1. **Describe crime patterns** with frequency analysis, conditional probability, mean and variance, and correlation.
2. **Predict crime hotspots.** Is a place (community area) at a given weekday and time slot **high risk**? Three models do this: Logistic Regression (baseline), Decision Tree and Random Forest.
3. **Compare the models statistically** on the same held-out test set: accuracy with 95% confidence intervals, McNemar's test and a bootstrap CI for each pair. *Is a model really better, or could the difference be due to test-sample variation?*

The website has three pages:

- **Hotspot predictor:** choose a place and get its crime-pattern analysis, every model's predicted high-risk periods, a check for a specific day and time, and the riskiest places at that time.
- **Crime patterns:** city-wide statistics.
- **Model comparison:** the statistical comparison of the three models.


## Structure

```
app.py                  Streamlit website (loads data, trains models once, page menu)
main.py                 command-line run of the whole analysis -> outputs/
views/
  predictor.py          page: hotspot predictor for a chosen place
  patterns.py           page: crime-pattern statistics
  comparison.py         page: model comparison
  charts.py             Plotly chart helpers
src/
  config.py             settings, community-area names, crime categories
  data.py               [1] load & clean incidents; build place-time cells
  statistics.py         [2] frequency, conditional probability, mean/variance, correlation
  models.py             [3-4] 70/30 split, high-risk label, the 3 models
  evaluation.py         [5] accuracy + Wilson CI, McNemar test, bootstrap CI
  hotspot.py            [6] area profile, risk grid, riskiest areas
  figures.py            report figures
tests/test_statistics.py
data/chicago_crimes_sample.csv
report/                 Word / PDF report
```

```
app.py / main.py -> src/data.py -> src/statistics.py
                                -> src/models.py -> src/evaluation.py
                                                 -> src/hotspot.py
app.py -> views/*.py -> views/charts.py
```

## Results (seed 42)

| Model | Test accuracy (95% CI) | ROC-AUC |
|---|---|---|
| Logistic Regression | 76.0% (72.6–79.2) | 0.635 |
| Decision Tree | 91.3% (88.9–93.3) | 0.939 |
| Random Forest | 92.4% (90.1–94.2) | 0.957 |

- Decision Tree and Random Forest beat Logistic Regression by about 15–16 points. McNemar p < 10⁻¹⁷, so the difference is real.
- Random Forest vs Decision Tree: +1.1 points, with a 95% CI of −0.8 to +2.9 and McNemar p = 0.35. This difference **could be due to test-sample variation**.
