"""Small Plotly helpers used by the pages."""

import plotly.graph_objects as go

from src.config import BLUE, MODEL_COLORS, ORANGE

FONT = dict(family="Inter, Segoe UI, sans-serif", size=13, color="#2b2b2b")


def layout(fig, title=None, height=330, x=None, y=None, legend=False):
    fig.update_layout(title=dict(text=title, x=0, y=0.98, yanchor="top", font=dict(size=14)) if title else None,
                      height=height, font=FONT, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=(80 if legend else 45) if title else 15, b=10),
                      showlegend=legend,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    fig.update_xaxes(title=x, showgrid=False, linecolor="#e4e3df")
    fig.update_yaxes(title=y, gridcolor="#eeeeec", zeroline=False)
    return fig


def bars(labels, values, title, y=None, color=BLUE, fmt=".1%", horizontal=False, height=330):
    text = [format(v, fmt) for v in values]
    if horizontal:
        fig = go.Figure(go.Bar(y=list(labels)[::-1], x=list(values)[::-1], orientation="h",
                               marker_color=color, text=text[::-1], textposition="outside",
                               cliponaxis=False))
        fig.update_xaxes(range=[0, max(values) * 1.25])
    else:
        fig = go.Figure(go.Bar(x=list(labels), y=list(values), marker_color=color, text=text,
                               textposition="outside", cliponaxis=False))
    return layout(fig, title, height, y=y)


def grouped_bars(table, title, y=None, colors=(BLUE, ORANGE)):
    fig = go.Figure([go.Bar(x=table.index.astype(str), y=table[c], name=c, marker_color=col,
                            text=[f"{v:.0%}" for v in table[c]], textposition="outside",
                            cliponaxis=False)
                     for c, col in zip(table.columns, colors)])
    fig.update_layout(barmode="group")
    return layout(fig, title, y=y, legend=True)


def heatmap(table, title, fmt=".2f", colorscale="Blues", zmin=None, zmax=None, height=300,
            showscale=True):
    z = table.values.astype(float)
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in table.columns], y=[str(i) for i in table.index],
                               colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=showscale,
                               text=[[format(v, fmt) for v in r] for r in z], texttemplate="%{text}",
                               hovertemplate="%{y}, %{x}: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="white")
    return layout(fig, title, height)


def accuracy_ci(table, title):
    fig = go.Figure()
    for _, r in table.iterrows():
        fig.add_trace(go.Scatter(x=[r["Accuracy"] * 100], y=[r["Model"]], mode="markers",
                                 marker=dict(size=13, color=MODEL_COLORS[r["Model"]]),
                                 error_x=dict(type="data", symmetric=False, thickness=2.5, width=8,
                                              array=[(r["CI high"] - r["Accuracy"]) * 100],
                                              arrayminus=[(r["Accuracy"] - r["CI low"]) * 100])))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeec")
    return layout(fig, title, 230, x="Test accuracy (%) with 95% confidence interval")


def bootstrap_hist(d, lo, hi, title):
    fig = go.Figure(go.Histogram(x=d * 100, nbinsx=40, marker_color=BLUE, opacity=0.85))
    fig.add_vrect(x0=lo * 100, x1=hi * 100, fillcolor=BLUE, opacity=0.08, line_width=0)
    fig.add_vline(x=0, line=dict(color=ORANGE, width=2.5), annotation_text="no difference",
                  annotation_position="top left")
    return layout(fig, title, 300, x="Accuracy difference (percentage points)", y="Resamples")


def area_map(c, value, title, height=520, highlight=None):
    """Community areas plotted at their centre (longitude, latitude), coloured by risk."""
    fig = go.Figure(go.Scatter(
        x=c["longitude"], y=c["latitude"], mode="markers",
        marker=dict(size=10 + 26 * c[value], color=c[value], colorscale="YlOrRd", cmin=0, cmax=1,
                    line=dict(color="white", width=1), colorbar=dict(title="P(high risk)")),
        text=[f"{a}<br>P(high risk) = {p:.2f}<br>observed incidents = {n}"
              for a, p, n in zip(c["area"], c[value], c["incidents"])],
        hoverinfo="text"))
    if highlight is not None:
        r = c[c["area"] == highlight].iloc[0]
        fig.add_trace(go.Scatter(x=[r["longitude"]], y=[r["latitude"]], mode="markers+text",
                                 marker=dict(symbol="star", size=20, color="#1f2328"),
                                 text=[highlight], textposition="top center", hoverinfo="skip"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1.34, showgrid=False, showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return layout(fig, title, height, x="west ← → east", y="south ← → north")
