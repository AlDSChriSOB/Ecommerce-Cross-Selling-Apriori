"""Original Olist data loader and leakage-safe chronological basket construction."""
from pathlib import Path
import pandas as pd

REQUIRED = (
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv",
)

def load_olist(raw_dir: str | Path) -> pd.DataFrame:
    """Join order, item and product tables at the order/product grain.

    Only delivered orders with valid purchase timestamps and known categories are
    used. No customer/payment/review joins: those would multiply order-item rows.
    """
    raw_dir = Path(raw_dir)
    missing = [name for name in REQUIRED if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing original Olist CSV files in {raw_dir}: {missing}. "
            "Download https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
        )
    orders = pd.read_csv(raw_dir / REQUIRED[0], usecols=[
        "order_id", "order_status", "order_purchase_timestamp"])
    items = pd.read_csv(raw_dir / REQUIRED[1], usecols=[
        "order_id", "product_id", "order_item_id", "price"])
    products = pd.read_csv(raw_dir / REQUIRED[2], usecols=[
        "product_id", "product_category_name"])
    translation = pd.read_csv(raw_dir / REQUIRED[3], usecols=[
        "product_category_name", "product_category_name_english"])

    for key, frame in (("order_id", orders), ("product_id", products),
                       ("product_category_name", translation)):
        if frame[key].dropna().duplicated().any():
            raise ValueError(f"Unexpected duplicate reference key: {key}")

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce")
    orders = orders.loc[
        orders["order_status"].eq("delivered") &
        orders["order_purchase_timestamp"].notna(),
        ["order_id", "order_purchase_timestamp"]]
    merged = items.merge(orders, on="order_id", how="inner", validate="many_to_one")
    merged = merged.merge(products, on="product_id", how="left", validate="many_to_one")
    merged = merged.merge(translation, on="product_category_name", how="left",
                          validate="many_to_one")
    merged["category"] = merged["product_category_name_english"].fillna(
        merged["product_category_name"])
    merged = merged.dropna(subset=["product_id", "category"])
    merged = merged.loc[merged["price"] > 0].copy()
    return merged[["order_id", "order_purchase_timestamp", "product_id",
                   "category", "price"]]

def build_baskets(rows: pd.DataFrame, level: str = "category") -> pd.DataFrame:
    """One order per row; items are unique, sorted tuples.

    The category analysis is primary because the product catalogue is sparse.
    Product-level analysis is provided only as a feasibility sensitivity check.
    """
    if level not in {"category", "product_id"}:
        raise ValueError("level must be category or product_id")
    frame = rows.dropna(subset=["order_id", "order_purchase_timestamp", level])
    result = frame.groupby("order_id", as_index=False).agg(
        timestamp=("order_purchase_timestamp", "min"),
        items=(level, lambda x: tuple(sorted(set(map(str, x))))),
    )
    result = result.loc[result["items"].map(len).gt(0)].copy()
    return result.sort_values(["timestamp", "order_id"]).reset_index(drop=True)

def chronological_split(baskets: pd.DataFrame, train_share: float = .6,
                        validation_share: float = .2):
    """Disjoint 60/20/20 timestamp-ordered partitions, including timestamp ties."""
    if not 0 < train_share < 1 or not 0 < validation_share < 1 - train_share:
        raise ValueError("Require positive train, validation and test fractions.")
    frame = baskets.sort_values(["timestamp", "order_id"]).reset_index(drop=True)
    if len(frame) < 5:
        raise ValueError("At least five orders required for chronological splitting.")
    cut_train = frame.iloc[int(len(frame) * train_share)]["timestamp"]
    cut_test = frame.iloc[int(len(frame) * (train_share + validation_share))]["timestamp"]
    train = frame.loc[frame["timestamp"] < cut_train].copy()
    valid = frame.loc[
        frame["timestamp"].ge(cut_train) & frame["timestamp"].lt(cut_test)].copy()
    test = frame.loc[frame["timestamp"].ge(cut_test)].copy()
    if any(x.empty for x in (train, valid, test)):
        raise ValueError("Not enough distinct timestamps for disjoint splits.")
    return train, valid, test
