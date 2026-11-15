# Project 02 — Executed Olist empirical findings

**Source:** original publicly released Olist Brazilian E-Commerce Public Dataset, downloaded by the reproducible GitHub Actions research workflow on 15 November 2026.

**Verified execution:** [GitHub Actions notebook-and-experiment run](https://github.com/AlDSChriSOB/Ecommerce-Cross-Selling-Apriori/actions/runs/43776138034).

## Dataset and feasibility audit

- Delivered orders with timestamps and valid positive-price, categorised items: **95,146** orders (108,660 joined item rows).
- Unique product identifiers: **31,632**; unique categories: **73**.
- Orders containing multiple distinct products: **3,141** (3.30%).
- Orders containing multiple distinct categories: **723** (0.76%).
- Training/validation/test orders: **57,087 / 19,029 / 19,030**; only **181** test orders were eligible for hidden-category evaluation.

## Test performance — paired masked category task

| Method | Hit@5 | Hit@10 | Rule-trigger rate |
|---|---:|---:|---:|
| Popularity | 45.30% | 60.22% | N/A |
| Co-occurrence | 52.49% | 68.51% | N/A |
| Apriori + popularity fallback | 45.30% | 60.22% | 1.10% |

**Paired bootstrap comparison:** Apriori minus co-occurrence Hit@10 = **-8.29 percentage points**, 95% paired percentile CI **[-14.92, -2.21] pp** (10,000 invoice-resampled replicates). This interval excludes zero *conditional on this held-out sample, the model selection and the single masking seed*. It does not establish external generalisation.

The validation step selected **minimum count 20**, **minimum confidence 0.1**, leading to **1** retained training association rule(s). Apriori rules triggered predictions on only **2 of 181** eligible test orders. The apparent Apriori performance largely reflects popularity fallback.

## Interpretation

1. **Negative finding:** This dataset and the prespecified category-level Apriori configuration do not demonstrate that association rules outperform simpler co-occurrence signals. Co-occurrence has a higher held-out Hit@10 for this split.
2. **Sparsity matters:** Only 0.76% of eligible historical orders have two or more categories. Even a popular category can dominate a single hidden category; the study is not evidence of strong SKU-level cross-sell opportunities.
3. **No business uplift has been measured:** Hit@K on held-out within-order categories is not next-purchase accuracy, attributable conversions, profit, causal revenue uplift, or a user experiment.
4. **Sample uncertainty and selection:** Only 181 multi-category test orders were scored, after restricting to delivered orders and valid categories. Wider evaluation across time splits and masking seeds is a prerequisite for a publication claim.

## Reproducibility

The following are **actual machine-generated outputs**, committed after a successful GitHub Actions run:

- [Executed Jupyter notebook](../notebooks/01_olist_research_EXECUTED.ipynb)
- [Metrics and validation grid](../results/metrics.json)
- [Per-order paired outcomes](../results/per_order_hits.csv)
- [Frequent-itemset association rules](../results/association_rules.csv)
- [Performance figure](../results/figures/hit_rate.png)

The source Olist CSV data are not redistributed in this repository. The tests use small hand-constructed *correctness fixtures*; no synthetic data were used to generate the empirical findings.
