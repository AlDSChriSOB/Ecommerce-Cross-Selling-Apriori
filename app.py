"""Read-only Streamlit dashboard for *executed*, real-data research results."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Olist Cross-Selling Research", layout="wide")
st.title("Olist e-commerce | Apriori cross-selling")
st.caption("Project 02 — observed historical dataset, chronological basket-completion benchmark.")
file = Path("results/metrics.json")
if not file.exists():
    st.info(
        "No empirical results have been generated yet. Download the original Olist "
        "dataset and run: python -m scripts.run_research --data-dir data/raw "
        "--output results. The app intentionally does not fabricate metrics."
    )
    st.markdown("[Original dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)")
    st.stop()

with file.open(encoding="utf-8") as handle:
    payload = json.load(handle)
audit = payload["audit"]
a, b, c = st.columns(3)
a.metric("Eligible held-out orders", f"{audit['test_eligible_baskets']:,}")
b.metric("Multi-category share", f"{audit['multi_category_rate']:.1%}")
c.metric("Modelled categories", f"{audit['distinct_categories']:,}")

metrics = pd.DataFrame(payload["test_metrics"])
st.subheader("Test-set recommendation performance")
st.dataframe(metrics, use_container_width=True)
st.bar_chart(metrics.pivot(index="k", columns="method", values="hit_rate_at_k"))
st.subheader("Paired bootstrap comparison")
st.json(payload["paired_bootstrap"])
st.subheader("Methodological limitations")
for item in payload["limitations"]:
    st.write("- " + item)
