"""A transparent Apriori frequent-itemset miner with singleton-consequent rules."""
from collections import Counter
from itertools import combinations
from math import log1p

def frequent_itemsets(baskets, min_count: int = 10, max_len: int = 3):
    """Count frequent itemsets, applying Apriori subset pruning at each level."""
    if min_count < 1 or max_len < 1:
        raise ValueError("min_count and max_len must be positive")
    tx = [frozenset(items) for items in baskets]
    if not tx:
        raise ValueError("Cannot fit on empty training data")
    singles = Counter((v,) for basket in tx for v in basket)
    counts = {k: v for k, v in singles.items() if v >= min_count}
    previous = set(counts)
    for k in range(2, max_len + 1):
        counts_k = Counter()
        for basket in tx:
            active = sorted(v for v in basket if (v,) in counts)
            for itemset in combinations(active, k):
                if all(sub in previous for sub in combinations(itemset, k - 1)):
                    counts_k[itemset] += 1
        previous = {itemset for itemset, num in counts_k.items()
                    if num >= min_count}
        if not previous:
            break
        counts.update({c: counts_k[c] for c in previous})
    return counts, len(tx)

def generate_rules(counts: dict, n_orders: int,
                   min_confidence: float = .05, min_lift: float = 1.0):
    """Produce antecedents of size 1–2 and singleton consequents.

    This is full Apriori itemset enumeration up to max_len in frequent_itemsets,
    with only singleton-consequent recommendation rules retained.
    """
    if n_orders <= 0:
        raise ValueError("n_orders must be positive")
    rules = []
    for itemset, joint in counts.items():
        if len(itemset) < 2:
            continue
        for target in itemset:
            antecedent = tuple(x for x in itemset if x != target)
            base = counts.get(antecedent, 0)
            target_count = counts.get((target,), 0)
            if not base or not target_count:
                continue
            confidence = joint / base
            lift = confidence / (target_count / n_orders)
            if confidence >= min_confidence and lift >= min_lift:
                rules.append({
                    "antecedent": antecedent, "consequent": target,
                    "support": joint / n_orders, "support_count": joint,
                    "confidence": confidence, "lift": lift,
                    "leverage": joint / n_orders -
                                (base / n_orders) * (target_count / n_orders),
                })
    return sorted(rules, key=lambda r: (
        -r["confidence"], -r["lift"], -r["support_count"],
        r["antecedent"], r["consequent"]))

def top_rules(rules, context, k=10):
    """Highest-scoring matching consequent per unseen item, with no fallback."""
    observed = set(context)
    scores = {}
    for rule in rules:
        if set(rule["antecedent"]).issubset(observed):
            product = rule["consequent"]
            if product not in observed:
                score = rule["confidence"] * log1p(rule["lift"])
                scores[product] = max(score, scores.get(product, float("-inf")))
    return sorted(scores, key=lambda item: (-scores[item], item))[:k]
