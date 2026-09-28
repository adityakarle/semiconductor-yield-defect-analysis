"""Streamlit dashboard: yield trends and defect drivers over the SECOM line."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# `streamlit run app/dashboard.py` doesn't add the project root to sys.path
# (unlike `python -m streamlit run ...`), so the src package import fails
# without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASET_PATH, FEATURE_IMPORTANCE_PATH, METRICS_PATH

st.set_page_config(page_title="Semiconductor Yield & Defect Analysis", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_parquet(DATASET_PATH)
    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return df, importance, metrics


REQUIRED_ARTIFACTS = [DATASET_PATH, FEATURE_IMPORTANCE_PATH, METRICS_PATH]

if not all(path.exists() for path in REQUIRED_ARTIFACTS):
    # Processed artifacts are gitignored, so a fresh deploy (e.g. Streamlit
    # Community Cloud) won't have them yet — or an earlier run may have been
    # interrupted partway, leaving some artifacts but not others. Either way,
    # the SECOM download + pipeline run only takes a few seconds, so bootstrap
    # it here instead of erroring out.
    with st.spinner("First run: downloading SECOM data and training the model..."):
        from src.pipeline import run_pipeline

        run_pipeline()

df, importance, metrics = load_data()

st.title("Semiconductor Manufacturing Yield & Defect Analysis")
st.caption("Real sensor data from the UCI SECOM dataset — an actual semiconductor fab line.")

# --- KPI row ---------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Units", f"{metrics['n_units']:,}")
col2.metric("Fail rate", f"{metrics['fail_rate'] * 100:.1f}%")
col3.metric("Sensors kept / raw", f"{metrics['n_sensors_after_cleaning']} / {metrics['n_sensors_raw']}")
best = next(r for r in metrics["model_comparison"] if r["model"] == metrics["best_model"])
col4.metric(f"Best model PR-AUC ({metrics['best_model']})", f"{best['pr_auc']:.3f}")

st.divider()

# --- Yield trend -------------------------------------------------------------
st.subheader("Yield trend over time")
trend = df[["timestamp", "fail"]].sort_values("timestamp").copy()
trend["date"] = trend["timestamp"].dt.date
daily = trend.groupby("date").agg(units=("fail", "size"), fails=("fail", "sum")).reset_index()
daily["fail_rate"] = daily["fails"] / daily["units"]
daily["rolling_fail_rate"] = daily["fail_rate"].rolling(7, min_periods=1).mean()

fig_trend = go.Figure()
fig_trend.add_trace(go.Bar(x=daily["date"], y=daily["units"], name="Units/day", yaxis="y2", opacity=0.25))
fig_trend.add_trace(go.Scatter(x=daily["date"], y=daily["rolling_fail_rate"], name="7-day rolling fail rate", mode="lines"))
fig_trend.update_layout(
    yaxis=dict(title="Fail rate", tickformat=".0%"),
    yaxis2=dict(title="Units/day", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=400,
)
st.plotly_chart(fig_trend, width="stretch")

st.divider()

# --- Feature importance ------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("Sensors most predictive of failure")
    top_n = st.slider("Show top N sensors", min_value=5, max_value=40, value=15)
    top_features = importance.head(top_n).sort_values("importance")
    fig_imp = px.bar(top_features, x="importance", y="sensor", orientation="h")
    fig_imp.update_layout(height=max(350, top_n * 22))
    st.plotly_chart(fig_imp, width="stretch")

with right:
    st.subheader("Model comparison (PR-AUC)")
    comp = pd.DataFrame(metrics["model_comparison"])
    st.dataframe(
        comp[["model", "pr_auc", "roc_auc", "precision_fail", "recall_fail", "f1_fail"]].round(3),
        hide_index=True,
        width="stretch",
    )

    pr = metrics["best_pr_curve"]
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="PR curve"))
    fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=300, margin=dict(t=10))
    st.plotly_chart(fig_pr, width="stretch")

st.divider()

# --- Sensor drill-down --------------------------------------------------------
st.subheader("Sensor drill-down")
sensor = st.selectbox("Sensor", options=importance["sensor"].tolist())
fig_dist = px.histogram(
    df, x=sensor, color=df["fail"].map({0: "Pass", 1: "Fail"}),
    barmode="overlay", opacity=0.6, nbins=40,
    labels={"color": "Outcome"},
)
fig_dist.update_layout(height=350)
st.plotly_chart(fig_dist, width="stretch")
