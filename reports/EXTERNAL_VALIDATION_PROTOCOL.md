# External validation protocol and journal-readiness audit

## Status

The original Olist category-basket benchmark and 15 overlapping-data sensitivity runs are complete. An external methodological replication using the **real UCI Online Retail dataset (DOI: 10.24432/C5BW33)** has been implemented in `scripts/run_external_validation.py` and GitHub Actions. **Do not interpret external results as verified until the external workflow succeeds and the generated outputs are inspected.**

## Independent evidence design

**Source population:** UCI Online Retail, a UK non-store retailer's historical invoice-line transactions. It is independent of the Brazilian Olist marketplace.

**Non-equivalent prediction task:** SKU/StockCode masked-invoice completion rather than Olist product-category masked-order completion. This is an external **methodological replication** of association-rule usefulness, *not* a direct replication of the Olist category-level numerical effect.

**Eligibility:** non-cancelled invoices with positive quantities, positive prices, valid dates and stock codes; one unique SKU set per invoice.

**Chronology:** train/validation/test split of approximately 60/20/20, preserving timestamp ties. Hyperparameter selection only on validation orders; frozen test predictions.

**Apriori specification:** pairwise frequent itemsets (maximum itemset size 2) for tractable SKU-level mining; support counts {20, 50, 100}, minimum confidences {.02, .05, .10}, lift threshold >=1. The main Olist analysis permits itemsets up to size 3, another explicit difference.

**Baselines:** popularity, co-occurrence and Apriori with popularity fallback. Hide one uniformly sampled SKU from each multi-SKU test invoice; evaluate Hit@5 and Hit@10 on paired targets, plus rule-trigger coverage and a 10,000-replicate paired bootstrap interval for Apriori minus co-occurrence Hit@10.

**Primary external question:** Does Apriori with fallback outperform co-occurrence at Hit@10 on the independently sourced UK SKU-basket task?

**Interpretation:** A negative result on both datasets would strengthen a cross-domain caution about this implementation, but not establish a universal failure of Apriori. A positive external result would show task/data dependence and help refine the scientific contribution.

## Critical journal submission gates

| Gate | State | Required evidence |
|---|---|---|
| Original Olist data and executable pipeline | Complete | Executed notebook and workflow logs |
| Olist chronological test and paired uncertainty | Complete | metrics.json and per-order outcomes |
| Olist sensitivity across splits and masking seeds | Complete | 15-run CSV and summary |
| Independently sourced external dataset | In progress | Successful UCI workflow, audited source and outputs |
| External replication interpretation | Pending | Real executed external metrics and confidence intervals |
| Repeated independent researcher reproduction | Not complete | Independent machine/team replication of source-to-result pipeline |
| Broader contemporary literature review | Not complete | Verified references and comparison to recent recommenders |
| Threats to validity and external generalisation | Partly complete | Explicitly qualified across datasets and task differences |
| Manuscript and journal-specific formatting | Not complete | Full paper, venue requirements, author approval |
| Ethics, licensing and data availability statements | Partly complete | Check UCI/Olist terms, document no redistribution of original datasets |
| Reviewer-style audit | Not complete | External methodological critique and revisions |

## Research integrity

Do not claim causal revenue uplift, online recommendation improvement, statistically independent 15-fold evidence, or generalisation to all retail sectors. Do not label the UK study a direct replication of the Olist effect because the item granularity and maximum itemset sizes differ. Do not claim journal readiness until the above gates are addressed.

## Source links

- [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- [Olist robustness study](ROBUSTNESS_STUDY.md)
