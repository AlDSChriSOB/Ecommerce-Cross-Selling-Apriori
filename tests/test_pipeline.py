"""Deterministic correctness tests. Tiny hand-constructed fixtures are NOT research data."""
import pandas as pd
import numpy as np
from src.olist_recs.apriori import frequent_itemsets, generate_rules
from src.olist_recs.data import build_baskets, chronological_split
from src.olist_recs.evaluation import fit, recommend, masked_evaluation, summary, paired_bootstrap

def fixture_orders():
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    return pd.DataFrame({
        "order_id": [f"o{i:02d}" for i in range(12)],
        "timestamp": dates,
        "items": [
            ("a", "b", "c"), ("a", "b"), ("a", "c"), ("b", "c"),
            ("a", "b"), ("a", "b", "c"), ("a", "b"), ("a", "c"),
            ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c"),
        ]})

def test_apriori_known_itemsets_and_rules():
    baskets = [("a", "b", "c"), ("a", "b"), ("a",), ("b", "c")]
    counts, n = frequent_itemsets(baskets, min_count=2, max_len=3)
    assert n == 4
    assert counts[("a",)] == 3
    assert counts[("a", "b")] == 2
    assert ("a", "b", "c") not in counts
    rules = generate_rules(counts, n, min_confidence=.5)
    assert any(r["antecedent"] == ("a",) and r["consequent"] == "b"
               and abs(r["confidence"] - 2 / 3) < 1e-12 for r in rules)

def test_basket_unique_items_and_chronology():
    frame = pd.DataFrame({
        "order_id": ["first", "first", "first", "second", "second"],
        "order_purchase_timestamp": pd.to_datetime(
            ["2020-01-01"] * 3 + ["2020-01-02"] * 2),
        "category": ["b", "a", "a", "b", "c"],
        "product_id": ["1", "2", "2", "3", "4"],
    })
    baskets = build_baskets(frame)
    assert baskets.iloc[0]["items"] == ("a", "b")
    assert baskets.iloc[1]["items"] == ("b", "c")

def test_temporal_split_never_shares_timestamp():
    orders = fixture_orders()
    train, val, test = chronological_split(orders)
    assert len(train) + len(val) + len(test) == len(orders)
    assert train.timestamp.max() < val.timestamp.min()
    assert val.timestamp.max() < test.timestamp.min()
    assert not (set(train.order_id) & set(val.order_id) | set(train.order_id) & set(test.order_id))

def test_no_observed_recommendations_and_paired_mask():
    orders = fixture_orders()
    train = orders.iloc[:8]
    model = fit(train, min_count=2, min_confidence=.05)
    for method in ("popularity", "cooccurrence", "apriori"):
        ranks, _ = recommend(model, ("a",), method, k=10)
        assert "a" not in ranks
        assert len(ranks) == len(set(ranks))
    outcomes1 = masked_evaluation(model, orders.iloc[8:], seed=42)
    outcomes2 = masked_evaluation(model, orders.iloc[8:], seed=42)
    pd.testing.assert_frame_equal(outcomes1, outcomes2)
    assert len(summary(outcomes1)) == 6
    bootstrap = paired_bootstrap(outcomes1, replicates=150, seed=1)
    assert bootstrap["n"] == len(outcomes1)
    assert -1 <= bootstrap["ci95_percentile"][0] <= 1

def test_rules_use_only_fit_baskets():
    old = pd.DataFrame({"items": [("a", "b")] * 8})
    future = pd.DataFrame({"items": [("z", "x")] * 8})
    model = fit(old, min_count=2)
    assert "z" not in model["popularity"]
    rec, _ = recommend(model, ("a",), "apriori", 10)
    assert "z" not in rec
