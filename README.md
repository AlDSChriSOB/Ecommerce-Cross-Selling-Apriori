# Project 02 — E-Commerce Cross-Selling & Product Bundling Intelligence

**Intelligent Recommender Systems Research Lab | Apriori project 02 of 20**

An auditable investigation of product-category association rules for e-commerce basket completion, using the **original Olist Brazilian E-Commerce Public Dataset** (anonymised historical Brazilian orders, approximately 2016–2018).

**Research question:** *Do Apriori-derived association rules improve category-level masked-basket Hit Rate@K over popularity and co-occurrence baselines on chronologically later Olist orders?*

> **Research status (15 November 2026):** real Olist data downloaded and processed successfully in GitHub Actions; research runner, executed notebook and unit-test CI all completed. The observed outcome is **negative for Apriori relative to co-occurrence**, not an invented performance claim. See [evidence-grounded findings](reports/OBSERVED_FINDINGS.md) and the [executed notebook](notebooks/01_olist_research_EXECUTED.ipynb).

## Executed findings — original Olist data

| Measure | Observed result |
|---|---:|
| Valid delivered orders after cleaning | 95,146 |
| Multi-category orders | 723 (0.76%) |
| Eligible held-out test baskets | 181 |
| Popularity Hit Rate@10 | 60.22% |
| Co-occurrence Hit Rate@10 | **68.51%** |
| Apriori + popularity fallback Hit Rate@10 | 60.22% |
| Apriori rule-trigger rate | 1.10% |

A paired bootstrap for Apriori minus co-occurrence yields **−8.29 percentage points** (95% interval **−14.92 to −2.21 pp**), conditional on this split and masking seed. The category sparsity and near-absent Apriori rule coverage materially limit cross-sell claims.

**[Review all verified outputs](results/metrics.json)** · **[View figure](results/figures/hit_rate.png)** · **[Open executed notebook](notebooks/01_olist_research_EXECUTED.ipynb)**

## Robustness checks — completed

The original-data workflow has now executed **15 sensitivity evaluations**: three chronological splits × five masking seeds. Apriori + popularity fallback **did not outperform co-occurrence in any of the 15**; its average Hit@10 difference was **−9.50 percentage points** (range **−14.60 to −5.11 pp**). The average Apriori rule-trigger rate was only **0.76%**. Twelve of the 15 per-configuration paired bootstrap intervals excluded zero, but the evaluations overlap and are **not independent experiments**.

- [Read the complete robustness study](reports/ROBUSTNESS_STUDY.md)
- [Inspect the 15 observed results](results/robustness/split_seed_sensitivity.csv)
- [Inspect machine-readable robustness summary](results/robustness/summary.json)
- [View successful GitHub Actions execution](https://github.com/AlDSChriSOB/Ecommerce-Cross-Selling-Apriori/actions/runs/43776500009)

## Why category-level first?

Olist contains many unique product identifiers and many one-product orders. Rule mining on individual product IDs may be too sparse to support generalisable associations. This project **first audits both product-level and category-level multi-item baskets**, then uses category-level baskets as the primary feasible target. Any business recommendation must acknowledge that a category is not a concrete sellable SKU.

## Original dataset — no synthetic research data

**Provider:** Olist  
**Official source:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce  
**Required original CSVs** (kept out of Git):
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`
- `product_category_name_translation.csv`

Download the dataset from its source and extract these four files into `data/raw/`. The research does not require customer, review, payments, seller or geolocation tables. Confirm applicable licensing terms with the original publisher before redistributing data.

A manual GitHub Actions workflow, **Run real Olist research (manual)**, also attempts to acquire the official dataset using `kagglehub`, execute the research and provide downloadable workflow artifacts. Availability may depend on Kaggle access and service restrictions.

## Method

1. Join orders to line items, products and category translations; filter delivered orders, invalid timestamps, and nonpositive price lines; deduplicate within each order to unique categorical items.
2. Audit overall population, catalogue sizes, and multi-product versus multi-category basket rates.
3. Partition chronologically into approximately **60% training / 20% validation / 20% test** by purchase timestamp, keeping tied timestamps in the same interval.
4. Fit **true Apriori frequent-itemset enumeration** (maximum itemset size 3; min support as absolute order count), and derive 1- and 2-item antecedent rules with **single-category consequents**.
5. Tune `min_count ∈ {5, 10, 20}` and `min_confidence ∈ {.02, .05, .10}` exclusively on the validation period.
6. Refit on the full pre-test period with the frozen hyperparameters; evaluate on chronologically held-out orders with at least two distinct categories. Hide exactly one category per eligible order using a reproducible seed.
7. Compare **popularity**, **item co-occurrence**, and **Apriori rules with an explicit popularity fallback** on the *same hidden targets*.
8. Report Hit Rate@5/10, precision@5/10, nonempty recommendation coverage, *rule-trigger rate*, and paired 10,000-resample bootstrap confidence intervals for Apriori minus co-occurrence Hit Rate@10.

Training-only information governs all recommendations. Observed query items are excluded. The fallback is disclosed; a win caused solely by popularity fallback is not described as association-rule evidence.

## Run locally

```bash
git clone https://github.com/AlDSChriSOB/Ecommerce-Cross-Selling-Apriori.git
cd Ecommerce-Cross-Selling-Apriori
python -m pip install -r requirements.txt
# Extract original Olist CSV files to data/raw/ (see dataset link above)
python -m pytest -q
python -m scripts.run_research --data-dir data/raw --output results
jupyter notebook notebooks/01_olist_research.ipynb
streamlit run app.py
```

The runner creates:

```text
results/
  metrics.json
  per_order_hits.csv
  validation_grid.csv
  association_rules.csv
  RESEARCH_RESULTS.md
  figures/hit_rate.png
```

All outputs are regenerated from original data; this repository intentionally does **not** commit fabricated result numbers.

## Project layout

```text
src/olist_recs/data.py         Original table joins and chronological baskets
src/olist_recs/apriori.py      Frequent-itemset mining and association rules
src/olist_recs/evaluation.py   Baselines, paired masking, holdout, bootstrap
scripts/run_research.py       Validation tuning, frozen test, outputs
notebooks/01_olist_research.ipynb  Auditable, executable notebook
tests/test_pipeline.py        Small deterministic correctness fixtures (not research data)
app.py                        Displays only existing real-data results
.github/workflows/ci.yml      Automatic code/unit tests
.github/workflows/research.yml  Optional manual real-data experiment
```

## Limitations and research integrity

- Masked basket completion is **not** next-visit/next-order prediction; there is no causal test of added revenue or conversion uplift.
- Only delivered orders are retained; resulting selection bias must be disclosed.
- This historical marketplace sample cannot represent other geographies or current e-commerce.
- Singleton consequent rules from frequent itemsets of size ≤3 are *not* exhaustive multi-consequent association mining.
- Some model performance can arise from the popularity fallback; inspect trigger rates.
- Pairwise bootstrap on one test split quantifies uncertainty conditional on that split and mask seed.
- Future research should add rolling-origin evaluation, repeated masking seeds, product-level sensitivity when statistically feasible, price/profit-aware constraints and online experimentation.

## References

- Olist. [Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
- Agrawal, R. & Srikant, R. (1994). Fast algorithms for mining association rules in large databases. *VLDB*, 487–499.

Part of an independently developed portfolio research sequence, with Project 01 at [UK Retail Market-Basket Intelligence](https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori).
