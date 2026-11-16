"""Chronological, paired, masked-basket evaluation without future-data leakage."""
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from .apriori import frequent_itemsets, generate_rules, top_rules

METHODS = ("popularity", "cooccurrence", "apriori")

def fit(train: pd.DataFrame, min_count=10, min_confidence=.05, min_lift=1.0, max_len=3):
    baskets = list(train["items"])
    counts, n_orders = frequent_itemsets(baskets, min_count=min_count, max_len=max_len)
    rules = generate_rules(counts, n_orders,
                           min_confidence=min_confidence, min_lift=min_lift)
    popularity = Counter(item for basket in baskets for item in set(basket))
    pair_counts = Counter()
    for basket in baskets:
        for a, b in combinations(sorted(set(basket)), 2):
            pair_counts[(a, b)] += 1
    neighbors = {}
    for (a, b), count in pair_counts.items():
        neighbors.setdefault(a, []).append((b, count))
        neighbors.setdefault(b, []).append((a, count))
    return {
        "rules": rules, "popularity": popularity,
        "pairs": pair_counts, "neighbors": neighbors, "n_orders": n_orders,
        "frequent_itemsets": len(counts),
    }

def recommend(model, observed, method, k=10, fallback=True):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    seen = set(observed)
    popularity = model["popularity"]
    ranking = []
    rule_triggered = False
    if method == "apriori":
        ranking = top_rules(model["rules"], seen, k=k)
        rule_triggered = bool(ranking)
    elif method == "cooccurrence":
        scores = Counter()
        for a in seen:
            if a not in popularity:
                continue
            for b, count in model["neighbors"].get(a, ()):
                if b not in seen:
                    scores[b] += count / popularity[a]
        ranking = sorted(scores, key=lambda x: (-scores[x], x))[:k]
    if method == "popularity":
        ranking = sorted(
            (v for v in popularity if v not in seen),
            key=lambda v: (-popularity[v], v))[:k]
    if fallback and len(ranking) < k:
        for v in sorted(popularity, key=lambda x: (-popularity[x], x)):
            if v not in seen and v not in ranking:
                ranking.append(v)
                if len(ranking) == k:
                    break
    return ranking[:k], rule_triggered

def masked_evaluation(model, orders: pd.DataFrame, k_values=(5, 10), seed=42):
    """Mask exactly one item; pair all methods using the same hidden targets."""
    rng = np.random.default_rng(seed)
    rows = []
    for row in orders.itertuples(index=False):
        items = tuple(row.items)
        if len(items) < 2:
            continue
        hidden = items[int(rng.integers(len(items)))]
        context = tuple(item for item in items if item != hidden)
        outcome = {
            "order_id": row.order_id, "hidden": hidden,
            "target_seen_in_train": int(hidden in model["popularity"]),
        }
        for method in METHODS:
            ranked, triggered = recommend(model, context, method, max(k_values))
            outcome[f"{method}_rule_triggered"] = int(triggered)
            outcome[f"{method}_any_recommendation"] = int(bool(ranked))
            for k in k_values:
                outcome[f"{method}_hit_at_{k}"] = int(hidden in ranked[:k])
                outcome[f"{method}_returned_at_{k}"] = min(k, len(ranked))
        rows.append(outcome)
    if not rows:
        raise ValueError("No multi-category baskets available for evaluation")
    return pd.DataFrame(rows)

def summary(hits: pd.DataFrame, k_values=(5, 10)):
    n = len(hits)
    if n == 0:
        raise ValueError("Empty evaluation frame")
    result = []
    for method in METHODS:
        for k in k_values:
            column = hits[f"{method}_hit_at_{k}"]
            result.append({
                "method": method, "k": k, "eligible_baskets": n,
                "hits": int(column.sum()), "hit_rate_at_k": float(column.mean()),
                "precision_at_k": float(
                    (column / k).mean()),  # one relevant target; fixed K denominator
                "nonempty_recommendation_rate": float(
                    hits[f"{method}_any_recommendation"].mean()),
                "rule_trigger_rate": float(
                    hits[f"{method}_rule_triggered"].mean()),
            })
    return result

def paired_bootstrap(hits, method_a="apriori", method_b="cooccurrence",
                     k=10, replicates=10000, seed=20261115):
    """Paired invoice-resampled CI; no claims of causal sales uplift."""
    a = hits[f"{method_a}_hit_at_{k}"].to_numpy(dtype=np.float64)
    b = hits[f"{method_b}_hit_at_{k}"].to_numpy(dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Need paired outcomes from at least two invoices")
    diff = a - b
    rng = np.random.default_rng(seed)
    # Batched resampling bounds memory consumption on large holdouts.
    means = []
    remain = replicates
    while remain:
        count = min(500, remain)
        indices = rng.integers(0, len(diff), size=(count, len(diff)))
        means.extend(diff[indices].mean(axis=1).tolist())
        remain -= count
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "method_a": method_a, "method_b": method_b, "k": k,
        "n": len(diff), "difference": float(diff.mean()),
        "ci95_percentile": [float(low), float(high)],
        "replicates": replicates, "seed": seed,
        "interval_crosses_zero": bool(low <= 0 <= high),
    }
