# Observed Olist research results

This report is generated only by executing against original Olist CSVs.

Audited orders: 95,146

Eligible held-out orders: 181

Selected hyperparameters: {'min_count': 20, 'min_confidence': 0.1, 'validation_hit_rate_at_10': 0.5688622754491018, 'validation_rule_trigger_rate': 0.011976047904191617, 'rule_count': 1}

Paired bootstrap result: {'method_a': 'apriori', 'method_b': 'cooccurrence', 'k': 10, 'n': 181, 'difference': -0.08287292817679558, 'ci95_percentile': [-0.14917127071823205, -0.022099447513812154], 'replicates': 10000, 'seed': 20261115, 'interval_crosses_zero': False}

The result is a masked within-order category completion study, not a causal estimate of sales improvement.
