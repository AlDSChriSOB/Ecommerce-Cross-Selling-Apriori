"""External methodological replication on UCI Online Retail (UK, 2010–2011).

Independent dataset and domain. Product IDs are SKUs, unlike Olist categories.
This does NOT constitute direct validation of Olist category-level estimates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.olist_recs.data import build_baskets, chronological_split
from src.olist_recs.evaluation import fit, masked_evaluation, summary, paired_bootstrap

def load_uci(path):
    df = pd.read_excel(path, engine="openpyxl", dtype={"InvoiceNo": str, "StockCode": str})
    raw_n = len(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["InvoiceNo"] = df["InvoiceNo"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")
    mask = (
        df["InvoiceNo"].notna() &
        ~df["InvoiceNo"].str.startswith("C", na=False) &
        df["StockCode"].notna() &
        df["InvoiceDate"].notna() &
        (pd.to_numeric(df["Quantity"], errors="coerce") > 0) &
        (pd.to_numeric(df["UnitPrice"], errors="coerce") > 0)
    )
    clean = df.loc[mask].copy()
    clean = clean.rename(columns={
        "InvoiceNo": "order_id", "InvoiceDate": "order_purchase_timestamp",
        "StockCode": "product_id",
    })
    clean["product_id"] = clean["product_id"].astype(str)
    return clean, {"original_lines": raw_n, "positive_non_cancelled_lines": len(clean)}

def run(input_file="data/external/Online Retail.xlsx",
        output="results/external_validation", seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame, audit = load_uci(input_file)
    baskets = build_baskets(frame, level="product_id")
    audit.update({
        "unique_invoices": len(baskets),
        "unique_stock_codes": int(frame["product_id"].nunique()),
        "multi_sku_invoices": int(baskets["items"].map(len).ge(2).sum()),
        "multi_sku_share": float(baskets["items"].map(len).ge(2).mean()),
    })
    train, validation, test = chronological_split(baskets)
    audit.update({"train": len(train), "validation": len(validation), "test": len(test)})
    # Large UK invoices make triple enumeration impractical; this is explicitly
    # a pair-only Apriori replication, not Olist's max-length-three pipeline.
    candidates = []
    for support in (20, 50, 100):
        for conf in (.02, .05, .1):
            model = fit(train, min_count=support, min_confidence=conf, max_len=2)
            hits = masked_evaluation(model, validation, seed=seed)
            candidates.append({
                "min_count": support, "min_confidence": conf,
                "validation_apriori_hit10": float(hits["apriori_hit_at_10"].mean()),
                "validation_rule_trigger_rate": float(
                    hits["apriori_rule_triggered"].mean()),
            })
    chosen = sorted(candidates, key=lambda x: (
        -x["validation_apriori_hit10"], -x["min_count"],
        -x["min_confidence"]))[0]
    pretest = pd.concat([train, validation], ignore_index=True)
    model = fit(pretest, min_count=chosen["min_count"],
                min_confidence=chosen["min_confidence"], max_len=2)
    hits = masked_evaluation(model, test, seed=seed)
    if len(hits) < 100:
        raise ValueError("External dataset has too few eligible test baskets")
    boot = paired_bootstrap(hits, replicates=10000, seed=20261116)
    payload = {
        "source": "UCI Online Retail dataset (DOI:10.24432/C5BW33)",
        "independence": "Distinct UK non-store retail dataset, not Olist.",
        "non_equivalence": "StockCode/SKU basket completion with pair-only Apriori; "
                           "not directly comparable to Olist category-level task.",
        "audit": audit,
        "selected_hyperparameters": chosen,
        "validation_grid": candidates,
        "test_metrics": summary(hits),
        "paired_bootstrap": boot,
        "rule_count": len(model["rules"]),
        "cautions": [
            "External dataset is not an independent Olist population replicate.",
            "SKU-level item universe and pair-only Apriori differ from Olist categories.",
            "This evaluates one masked SKU from the same invoice, not a future sale.",
            "The original UK data describe a non-store gift retailer, not a supermarket.",
            "The model does not estimate causal commercial uplift.",
        ],
    }
    (output/"metrics.json").write_text(json.dumps(payload, indent=2)+"\n")
    pd.DataFrame(candidates).to_csv(output/"validation_grid.csv", index=False)
    hits.to_csv(output/"per_invoice_hits.csv", index=False)
    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="data/external/Online Retail.xlsx")
    parser.add_argument("--output", default="results/external_validation")
    args = parser.parse_args()
    result = run(args.input_file, args.output)
    print(json.dumps({
        "audit": result["audit"],
        "selected_hyperparameters": result["selected_hyperparameters"],
        "test_metrics": result["test_metrics"],
        "paired_bootstrap": result["paired_bootstrap"]}, indent=2))
