from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


SOURCE_FEATURES = [
    "search",
    "cat",
    "searches",
    "has_search_to_cart",
    "has_search_to_ord",
    "has_cat_to_cart",
    "has_cat_to_ord",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "to_cart",
    "to_ord",
    "gmv",
]


def _build_feature_names(source_features: list[str] | tuple[str, ...]) -> list[str]:
    names = ["dow", "days_seen", "week_idx"]
    for source in source_features:
        names.extend(
            [
                f"{source}_yesterday",
                f"{source}_sum3",
                f"{source}_sum7",
                f"{source}_mean7",
                f"{source}_active_days7",
                f"{source}_exp3",
                f"{source}_last_active",
                f"{source}_days_since_last_active",
            ]
        )
    names.extend(
        [
            "orders_hist_mean",
            "orders_hist_purchased_days",
            "carts_hist_mean",
            "any_activity_yesterday",
            "any_activity_sum7",
            "any_activity_mean7",
            "any_activity_active_days7",
            "days_since_last_any_activity",
            "to_ord_ewm7",
            "to_ord_ewm28",
            "to_ord_ewm_gap_7_28",
            "to_cart_ewm7",
            "to_cart_ewm28",
            "to_cart_ewm_gap_7_28",
            "any_activity_ewm7",
            "any_activity_ewm28",
            "any_activity_ewm_gap_7_28",
            "days_since_last_order",
            "days_since_last_cart",
            "days_since_last_search",
            "days_since_last_cat",
            "ord_per_cart_7",
            "search_to_cart_rate_7",
            "search_to_ord_rate_7",
            "cat_to_cart_rate_7",
            "cat_to_ord_rate_7",
        ]
    )
    return names


GBDT_FEATURE_NAMES = _build_feature_names(SOURCE_FEATURES)


def _activity_array(block: dict[str, np.ndarray], source_features: list[str] | tuple[str, ...]) -> np.ndarray:
    stacked = np.column_stack([np.asarray(block[name], dtype=float) for name in source_features])
    return (stacked.sum(axis=1) > 0).astype(float)


def _ewm_mean(hist: np.ndarray, span: float) -> float:
    hist = np.asarray(hist, dtype=float)
    if hist.size == 0:
        return 0.0
    alpha = 2.0 / (float(span) + 1.0)
    decay = (1.0 - alpha) ** np.arange(hist.size - 1, -1, -1, dtype=float)
    return float(np.dot(hist, decay) / np.maximum(decay.sum(), 1e-8))


@dataclass
class GBDTFeatureTable:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    row_index_train: pd.DataFrame
    row_index_test: pd.DataFrame


def build_feature_vector_for_day(
    block: dict[str, np.ndarray],
    source_features: list[str] | tuple[str, ...],
    activity_full: np.ndarray,
    dows: np.ndarray,
    day_idx: int,
) -> list[float]:
    feats: list[float] = [
        float(dows[day_idx]),
        float(day_idx),
        float(day_idx // 7),
    ]

    for name in source_features:
        arr = block[name]
        hist = arr[:day_idx]
        feats.append(float(arr[day_idx - 1]) if day_idx >= 1 else 0.0)
        feats.append(float(hist[-3:].sum()) if hist.size else 0.0)
        feats.append(float(hist[-7:].sum()) if hist.size else 0.0)
        feats.append(float(hist[-7:].mean()) if hist.size else 0.0)
        feats.append(float((hist[-7:] > 0).sum()) if hist.size else 0.0)
        if hist.size:
            weights = np.exp(-np.arange(hist.size - 1, -1, -1) / 3.0)
            feats.append(float(np.dot(hist, weights)))
            nz_idx = np.flatnonzero(hist > 0)
            if nz_idx.size:
                last_idx = int(nz_idx[-1])
                feats.append(float(hist[last_idx]))
                feats.append(float(day_idx - 1 - last_idx))
            else:
                feats.extend([0.0, float(day_idx)])
        else:
            feats.extend([0.0, 0.0, float(day_idx)])

    orders_hist = block["to_ord"][:day_idx]
    carts_hist = block["to_cart"][:day_idx]
    searches_hist = block["searches"][:day_idx]
    cat_hist = block["cat"][:day_idx]
    activity_hist = activity_full[:day_idx]

    feats.extend(
        [
            float(orders_hist.mean()) if orders_hist.size else 0.0,
            float((orders_hist > 0).sum()) if orders_hist.size else 0.0,
            float(carts_hist.mean()) if carts_hist.size else 0.0,
            float(activity_hist[-1]) if activity_hist.size else 0.0,
            float(activity_hist[-7:].sum()) if activity_hist.size else 0.0,
            float(activity_hist[-7:].mean()) if activity_hist.size else 0.0,
            float((activity_hist[-7:] > 0).sum()) if activity_hist.size else 0.0,
            float(day_idx - 1 - np.flatnonzero(activity_hist > 0)[-1]) if np.any(activity_hist > 0) else float(day_idx),
            _ewm_mean(orders_hist, span=7),
            _ewm_mean(orders_hist, span=28),
        ]
    )
    feats.append(feats[-2] - feats[-1])
    feats.extend(
        [
            _ewm_mean(carts_hist, span=7),
            _ewm_mean(carts_hist, span=28),
        ]
    )
    feats.append(feats[-2] - feats[-1])
    feats.extend(
        [
            _ewm_mean(activity_hist, span=7),
            _ewm_mean(activity_hist, span=28),
        ]
    )
    feats.append(feats[-2] - feats[-1])
    feats.extend(
        [
            float(day_idx - 1 - np.flatnonzero(orders_hist > 0)[-1]) if np.any(orders_hist > 0) else float(day_idx),
            float(day_idx - 1 - np.flatnonzero(carts_hist > 0)[-1]) if np.any(carts_hist > 0) else float(day_idx),
            float(day_idx - 1 - np.flatnonzero(searches_hist > 0)[-1]) if np.any(searches_hist > 0) else float(day_idx),
            float(day_idx - 1 - np.flatnonzero(cat_hist > 0)[-1]) if np.any(cat_hist > 0) else float(day_idx),
        ]
    )

    to_cart_sum7 = feats[3 + source_features.index("to_cart") * 8 + 2]
    searches_sum7 = feats[3 + source_features.index("searches") * 8 + 2]
    cat_sum7 = feats[3 + source_features.index("cat") * 8 + 2]
    to_ord_sum7 = feats[3 + source_features.index("to_ord") * 8 + 2]
    search_to_cart_sum7 = feats[3 + source_features.index("search_to_cart") * 8 + 2]
    search_to_ord_sum7 = feats[3 + source_features.index("search_to_ord") * 8 + 2]
    cat_to_cart_sum7 = feats[3 + source_features.index("cat_to_cart") * 8 + 2]
    cat_to_ord_sum7 = feats[3 + source_features.index("cat_to_ord") * 8 + 2]

    denom_cart = max(to_cart_sum7, 1.0)
    denom_search = max(searches_sum7, 1.0)
    denom_cat = max(cat_sum7, 1.0)
    feats.extend(
        [
            float(to_ord_sum7 / denom_cart),
            float(search_to_cart_sum7 / denom_search),
            float(search_to_ord_sum7 / denom_search),
            float(cat_to_cart_sum7 / denom_cat),
            float(cat_to_ord_sum7 / denom_cat),
        ]
    )
    return feats


def build_feature_tables(
    full_df: pd.DataFrame,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    split_date: pd.Timestamp,
    target_col: str = "to_ord",
    source_features: list[str] | tuple[str, ...] | None = None,
) -> GBDTFeatureTable:
    source_features = list(source_features or SOURCE_FEATURES)
    train_rows: list[dict[str, float]] = []
    test_rows: list[dict[str, float]] = []
    train_index_rows: list[dict[str, object]] = []
    test_index_rows: list[dict[str, object]] = []

    grouped = full_df.sort_values(["user_id", "event_date"]).groupby("user_id", sort=False)
    for user_id, user_df in grouped:
        block = {"dates": user_df["event_date"].to_numpy()}
        for name in source_features:
            block[name] = user_df[name].to_numpy(dtype=float)
        dows = user_df["event_date"].dt.dayofweek.to_numpy(dtype=int)
        activity_full = _activity_array(block, source_features)
        target_arr = user_df[target_col].to_numpy(dtype=float)

        dates = user_df["event_date"].to_numpy(dtype="datetime64[ns]")
        analysis_mask = (dates >= np.datetime64(analysis_start)) & (dates <= np.datetime64(analysis_end))
        if not analysis_mask.any():
            continue
        idxs = np.flatnonzero(analysis_mask)
        for day_idx in idxs.tolist():
            feats = build_feature_vector_for_day(block, source_features, activity_full, dows, int(day_idx))
            row_index = {
                "user_id": int(user_id),
                "event_date": pd.Timestamp(user_df.iloc[day_idx]["event_date"]),
            }
            target = float(target_arr[day_idx])
            if row_index["event_date"] <= split_date:
                train_rows.append(feats)
                train_index_rows.append({**row_index, "target": target})
            else:
                test_rows.append(feats)
                test_index_rows.append({**row_index, "target": target})

    if not train_rows or not test_rows:
        raise ValueError("Empty train/test feature tables")

    feature_names = _build_feature_names(source_features)
    x_train = np.asarray(train_rows, dtype=np.float32)
    y_train = pd.DataFrame(train_index_rows)["target"].to_numpy(dtype=float)
    x_test = np.asarray(test_rows, dtype=np.float32)
    y_test = pd.DataFrame(test_index_rows)["target"].to_numpy(dtype=float)
    row_index_train = pd.DataFrame(train_index_rows)[["user_id", "event_date", "target"]]
    row_index_test = pd.DataFrame(test_index_rows)[["user_id", "event_date", "target"]]
    return GBDTFeatureTable(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_names=feature_names,
        row_index_train=row_index_train,
        row_index_test=row_index_test,
    )


def build_feature_panel(
    full_df: pd.DataFrame,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    target_col: str = "to_ord",
    source_features: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Build a feature row for every (user, day) in [analysis_start, analysis_end].

    Returns (x_panel, index_df, feature_names) where:
      * x_panel — float32 ndarray of shape (N, n_features),
      * index_df — DataFrame [user_id, event_date, target] of length N,
      * feature_names — list of column names matching x_panel columns.

    No train/test split is applied; callers slice index_df by date to
    construct any train/test windows they need.
    """
    source_features = list(source_features or SOURCE_FEATURES)
    rows: list[list[float]] = []
    user_ids: list[int] = []
    event_dates: list[pd.Timestamp] = []
    targets: list[float] = []

    grouped = full_df.sort_values(["user_id", "event_date"]).groupby("user_id", sort=False)
    for user_id, user_df in grouped:
        block = {"dates": user_df["event_date"].to_numpy()}
        for name in source_features:
            block[name] = user_df[name].to_numpy(dtype=float)
        dows = user_df["event_date"].dt.dayofweek.to_numpy(dtype=int)
        activity_full = _activity_array(block, source_features)
        target_arr = user_df[target_col].to_numpy(dtype=float)

        dates = user_df["event_date"].to_numpy(dtype="datetime64[ns]")
        analysis_mask = (dates >= np.datetime64(analysis_start)) & (dates <= np.datetime64(analysis_end))
        if not analysis_mask.any():
            continue
        idxs = np.flatnonzero(analysis_mask)
        ev = user_df["event_date"].to_numpy()
        for day_idx in idxs.tolist():
            feats = build_feature_vector_for_day(
                block, source_features, activity_full, dows, int(day_idx)
            )
            rows.append(feats)
            user_ids.append(int(user_id))
            event_dates.append(pd.Timestamp(ev[day_idx]))
            targets.append(float(target_arr[day_idx]))

    if not rows:
        raise ValueError("Empty feature panel")

    x_panel = np.asarray(rows, dtype=np.float32)
    index_df = pd.DataFrame(
        {"user_id": user_ids, "event_date": event_dates, "target": targets}
    )
    feature_names = _build_feature_names(source_features)
    return x_panel, index_df, feature_names


def fit_global_poisson_gbdt(
    feature_table: GBDTFeatureTable,
    seed: int = 42,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    max_iter: int = 200,
    min_samples_leaf: int = 40,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
    )
    model.fit(feature_table.x_train, feature_table.y_train)
    return model
