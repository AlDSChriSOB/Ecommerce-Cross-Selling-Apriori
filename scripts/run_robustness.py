"""Repeated masking and chronological-split sensitivity on original Olist orders.

This is a robustness analysis, not an independent validation population.
Hyperparameters are selected afresh using each split's validation interval.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.olist_recs.data import load_olist, build_baskets, chronological_split
from src.olist_recs.evaluation import fit, masked_evaluation, paired_bootstrap

def run(data_dir="data/raw", output="results/robustness",
        seeds=(11, 42, 101, 2026, 20261115),
        splits=((.50, .20), (.60, .20), (.70, .15))):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baskets = build_baskets(load_olist(data_dir), "category")
    records = []
    for train_share, validation_share in splits:
        train, validation, test = chronological_split(
            baskets, train_share=train_share, validation_share=validation_share)
        eligible = int(test["items"].map(len).ge(2).sum())
        if eligible < 30:
            raise RuntimeError(
                f"Only {eligible} eligible test orders for split "
                f"{train_share}/{validation_share}; cannot run.")
        pretest = pd.concat([train, validation], ignore_index=True)
        for seed in seeds:
            candidates = []
            for min_count in (5, 10, 20):
                for min_confidence in (.02, .05, .10):
                    candidate = fit(train, min_count=min_count,
                                    min_confidence=min_confidence)
                    val = masked_evaluation(candidate, validation, seed=seed)
                    candidates.append((
                        float(val["apriori_hit_at_10"].mean()),
                        min_count, min_confidence))
            # Match main experiment's deterministic tie-break exactly.
            best = sorted(candidates, key=lambda x: (-x[0], -x[1], -x[2]))[0]
            model = fit(pretest, min_count=best[1], min_confidence=best[2])
            hits = masked_evaluation(model, test, seed=seed)
            ci = paired_bootstrap(hits, replicates=2000, seed=20261115+seed)
            records.append({
                "train_share": train_share,
                "validation_share": validation_share,
                "test_share": round(1-train_share-validation_share, 5),
                "mask_seed": seed,
                "train_orders": len(train),
                "validation_orders": len(validation),
                "test_orders": len(test),
                "eligible_test_orders": len(hits),
                "min_count": best[1],
                "min_confidence": best[2],
                "rule_count": len(model["rules"]),
                "apriori_rule_trigger_rate": float(
                    hits["apriori_rule_triggered"].mean()),
                "popularity_hit_at_10": float(
                    hits["popularity_hit_at_10"].mean()),
                "cooccurrence_hit_at_10": float(
                    hits["cooccurrence_hit_at_10"].mean()),
                "apriori_hit_at_10": float(
                    hits["apriori_hit_at_10"].mean()),
                "apriori_minus_cooccurrence": ci["difference"],
                "paired_ci95_low": ci["ci95_percentile"][0],
                "paired_ci95_high": ci["ci95_percentile"][1],
            })
    df = pd.DataFrame(records)
    df.to_csv(output/"split_seed_sensitivity.csv", index=False)
    summary = {
        "dataset": "Original Olist Brazilian E-Commerce Public Dataset",
        "runs": len(df),
        "split_count": len(splits),
        "mask_seed_count": len(seeds),
        "apriori_minus_cooccurrence_mean": float(
            df["apriori_minus_cooccurrence"].mean()),
        "apriori_minus_cooccurrence_min": float(
            df["apriori_minus_cooccurrence"].min()),
        "apriori_minus_cooccurrence_max": float(
            df["apriori_minus_cooccurrence"].max()),
        "apriori_beats_cooccurrence_runs": int(
            (df["apriori_minus_cooccurrence"] > 0).sum()),
        "apriori_rule_trigger_rate_mean": float(
            df["apriori_rule_trigger_rate"].mean()),
        "limitations": [
            "The split windows overlap; results are not independent replicates.",
            "Multiple mask seeds reuse the same held-out orders.",
            "The percentile intervals are conditional on each split and mask seed.",
            "No sales uplift or next-order prediction is evaluated.",
            "The source contains very few multi-category baskets.",
        ],
    }
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="results/robustness")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output), indent=2))
