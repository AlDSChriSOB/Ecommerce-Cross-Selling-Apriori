"""Run an auditable Olist category-level Apriori experiment on the original CSVs.

Usage: python -m scripts.run_research --data-dir data/raw --output results
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.olist_recs.data import load_olist, build_baskets, chronological_split
from src.olist_recs.evaluation import fit, masked_evaluation, summary, paired_bootstrap


def run(data_dir="data/raw", output="results", seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_olist(data_dir)
    baskets = build_baskets(rows, level="category")
    product_baskets = build_baskets(rows, level="product_id")
    train, validation, test = chronological_split(baskets)

    audit = {
        "source": "Original Olist Brazilian E-Commerce Public Dataset",
        "rows_joined": int(len(rows)), "orders": int(len(baskets)),
        "distinct_categories": int(rows["category"].nunique()),
        "distinct_products": int(rows["product_id"].nunique()),
        "multi_category_orders": int(baskets["items"].map(len).ge(2).sum()),
        "multi_category_rate": float(baskets["items"].map(len).ge(2).mean()),
        "multi_product_orders": int(product_baskets["items"].map(len).ge(2).sum()),
        "multi_product_rate": float(product_baskets["items"].map(len).ge(2).mean()),
        "train_orders": int(len(train)), "validation_orders": int(len(validation)),
        "test_orders": int(len(test)), "mask_seed": seed,
    }
    if validation["items"].map(len).ge(2).sum() < 30:
        raise RuntimeError("Fewer than 30 eligible validation baskets; results too sparse.")
    if test["items"].map(len).ge(2).sum() < 100:
        raise RuntimeError("Fewer than 100 eligible test baskets; results too sparse.")

    # All parameter selection happens on the validation period, never on test.
    candidates = []
    for min_count in (5, 10, 20):
        for min_confidence in (.02, .05, .10):
            model = fit(train, min_count=min_count,
                        min_confidence=min_confidence)
            validation_hits = masked_evaluation(model, validation, seed=seed)
            hr10 = float(validation_hits["apriori_hit_at_10"].mean())
            candidates.append({
                "min_count": min_count,
                "min_confidence": min_confidence,
                "validation_hit_rate_at_10": hr10,
                "validation_rule_trigger_rate": float(
                    validation_hits["apriori_rule_triggered"].mean()),
                "rule_count": len(model["rules"]),
            })
    # Predefined deterministic tie-breaking: prefer larger support then confidence.
    chosen = sorted(candidates, key=lambda x: (
        -x["validation_hit_rate_at_10"], -x["min_count"],
        -x["min_confidence"]))[0]

    # Refit on the full pre-test period only, keeping the tuning parameters frozen.
    pretest = pd.concat([train, validation], ignore_index=True)
    model = fit(pretest, chosen["min_count"], chosen["min_confidence"])
    hits = masked_evaluation(model, test, seed=seed)
    metrics = summary(hits)
    bootstrap = paired_bootstrap(hits)
    audit["test_eligible_baskets"] = len(hits)
    audit["unseen_target_rate"] = float(1 - hits["target_seen_in_train"].mean())
    payload = {
        "audit": audit, "selected_hyperparameters": chosen,
        "validation_grid": candidates, "test_metrics": metrics,
        "paired_bootstrap": bootstrap, "trained_rule_count": len(model["rules"]),
        "frequent_itemsets": model["frequent_itemsets"],
        "limitations": [
            "Observational cross-sectional order baskets: no causal revenue uplift.",
            "One hidden category in the same order: not sequential next-order prediction.",
            "Delivered orders only; possible survivorship selection.",
            "Historical Olist marketplace is not representative of global e-commerce.",
            "Only one chronological holdout; repeat with alternative splits and mask seeds.",
            "Popularity fallback may account for hits when no association rule applies.",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    hits.to_csv(output / "per_order_hits.csv", index=False)
    pd.DataFrame(candidates).to_csv(output / "validation_grid.csv", index=False)
    rule_frame = pd.DataFrame([
        {"antecedent": " | ".join(r["antecedent"]), **{
            key: value for key, value in r.items() if key != "antecedent"}}
        for r in model["rules"]])
    rule_frame.to_csv(output / "association_rules.csv", index=False)

    metric_frame = pd.DataFrame(metrics)
    pivot = metric_frame.pivot(index="k", columns="method", values="hit_rate_at_k")
    ax = (pivot * 100).plot.bar(figsize=(8, 5), rot=0)
    ax.set(xlabel="Recommendation cutoff (K)", ylabel="Hit rate (%)",
           title="Olist category basket completion — chronological test")
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.savefig(figures / "hit_rate.png", dpi=220, bbox_inches="tight")
    plt.close()

    report = (
        "# Observed Olist research results\n\n"
        "This report is generated only by executing against original Olist CSVs.\n\n"
        f"Audited orders: {audit['orders']:,}\n\n"
        f"Eligible held-out orders: {len(hits):,}\n\n"
        f"Selected hyperparameters: {chosen}\n\n"
        f"Paired bootstrap result: {bootstrap}\n\n"
        "The result is a masked within-order category completion study, not a "
        "causal estimate of sales improvement.\n"
    )
    (output / "RESEARCH_RESULTS.md").write_text(report, encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = run(args.data_dir, args.output, args.seed)
    print(json.dumps({"audit": results["audit"],
                      "test_metrics": results["test_metrics"],
                      "paired_bootstrap": results["paired_bootstrap"]}, indent=2))
