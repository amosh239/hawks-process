import pandas as pd
from pathlib import Path
from .baselines import HourlySeasonalPoisson, HourOfWeekSeasonalPoisson, ScaledBaseline
from .hawkes import HawkesExp, PurchaseTargetHawkes
from .utils import temporal_train_test_split, ensure_dir


def calc_test_ll(
    model,
    train_times,
    test_times,
    train_types=None,
    test_types=None,
    window_start=None,
    train_end=None,
    full_end=None,
    denom=None,
):
    # Allow empty test windows (e.g., no purchases in test). In that case we still
    # can evaluate LL via the integral over (train_end, full_end], but we need full_end.
    if len(test_times) == 0 and full_end is None:
        return None

    full_times = train_times + test_times
    full_types = None
    train_types_use = None
    if train_types is not None and test_types is not None:
        train_types_use = train_types
        full_types = train_types + test_types

    ll_full = -model.nll(full_times, full_types, t_end=full_end or full_times[-1], window_start=window_start)
    ll_train = -model.nll(train_times, train_types_use, t_end=train_end or train_times[-1], window_start=window_start)
    denom = len(test_times) if denom is None else int(denom)
    if denom <= 0:
        return None
    return (ll_full - ll_train) / denom


def run_univariate_experiment(
    sequences,
    results_dir,
    train_ratio=0.8,
    min_events=20,
    max_sequences=None,
):
    rows = []
    selected = [s for s in sequences if len(s.get("times", [])) >= min_events]
    if max_sequences is not None:
        selected = selected[:max_sequences]

    for seq in selected:
        times = sorted(seq["times"])
        train_times, test_times, _, _ = temporal_train_test_split(times, train_ratio=train_ratio)
        if len(test_times) == 0:
            continue

        base = HourlySeasonalPoisson().fit(train_times)
        scaled = ScaledBaseline(base).fit(train_times)
        hawkes = HawkesExp(baseline=base).fit(train_times)

        ll_base = calc_test_ll(scaled, train_times, test_times)
        ll_hawkes = calc_test_ll(hawkes, train_times, test_times)
        if ll_base is None or ll_hawkes is None:
            continue

        rows.append(
            {
                "seq_id": seq.get("id"),
                "n_events": len(times),
                "ll_baseline": ll_base,
                "ll_hawkes": ll_hawkes,
                "ll_gain": ll_hawkes - ll_base,
                "mu": scaled.mu,
                "alpha": hawkes.params[1],
                "beta": hawkes.params[2],
            }
        )

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(results_dir)
    df.to_csv(out_dir / "summary.csv", index=False)
    if not df.empty:
        try:
            from .plots import plot_ll_gain_hist, plot_ll_vs_events

            plot_ll_gain_hist(df, out_dir / "ll_gain_hist.png")
            plot_ll_vs_events(df, out_dir / "ll_gain_vs_events.png")
        except Exception:
            pass
    return df


def run_purchase_experiment(
    sequences,
    results_dir,
    train_ratio=0.8,
    min_events=30,
    max_sequences=None,
    target_type=2,
    n_types=3,
    min_train_events=20,
    min_purchases_train=2,
):
    rows = []
    selected = [s for s in sequences if len(s.get("times", [])) >= min_events]
    if max_sequences is not None:
        selected = selected[:max_sequences]

    # Global seasonal profile: fit once on pooled train purchases across all users.
    # Per-user activity is handled by ScaledBaseline via MLE mu.
    pooled_purchases_train = []
    for seq in selected:
        times = sorted(seq["times"])
        types = [t for _, t in sorted(zip(seq["times"], seq["types"]))]

        train_times, test_times, train_types, test_types = temporal_train_test_split(times, types, train_ratio=train_ratio)
        if len(test_times) == 0:
            continue
        if len(train_times) < min_train_events:
            continue

        purchases_train = [t for t, tp in zip(train_times, train_types) if tp == target_type]
        if len(purchases_train) < min_purchases_train:
            continue
        pooled_purchases_train.extend(purchases_train)

    pooled_purchases_train = sorted(pooled_purchases_train)
    base = HourOfWeekSeasonalPoisson().fit(pooled_purchases_train)

    for seq in selected:
        times = sorted(seq["times"])
        types = [t for _, t in sorted(zip(seq["times"], seq["types"]))]

        train_times, test_times, train_types, test_types = temporal_train_test_split(times, types, train_ratio=train_ratio)
        if len(test_times) == 0:
            continue
        if len(train_times) < min_train_events:
            continue

        purchases_train = [t for t, tp in zip(train_times, train_types) if tp == target_type]
        purchases_test = [t for t, tp in zip(test_times, test_types) if tp == target_type]
        if len(purchases_train) < min_purchases_train:
            continue

        window_start = train_times[0]
        train_end = train_times[-1]
        full_end = test_times[-1]

        scaled = ScaledBaseline(base).fit(purchases_train, window_start=window_start, window_end=train_end)
        hawkes = PurchaseTargetHawkes(baseline=base, n_types=n_types, target_type=target_type).fit(train_times, train_types)

        # To avoid selection bias, we also evaluate sequences with 0 test purchases.
        # Use a time-normalized LL over the test window so denom is always defined.
        test_hours = max(1e-6, (full_end - train_end).total_seconds() / 3600.0)
        ll_base = calc_test_ll(
            scaled,
            purchases_train,
            purchases_test,
            window_start=window_start,
            train_end=train_end,
            full_end=full_end,
            denom=test_hours,
        )
        ll_hawkes = calc_test_ll(
            hawkes,
            train_times,
            test_times,
            train_types=train_types,
            test_types=test_types,
            window_start=window_start,
            train_end=train_end,
            full_end=full_end,
            denom=test_hours,
        )
        if ll_base is None or ll_hawkes is None:
            continue

        alphas = hawkes.params[1 : 1 + n_types]
        row = {
            "seq_id": seq.get("id"),
            "n_events": len(times),
            "train_events": len(train_times),
            "test_events": len(test_times),
            "purchases_train": len(purchases_train),
            "purchases_test": len(purchases_test),
            "test_hours": test_hours,
            "ll_unit": "per_hour",
            "ll_baseline": ll_base,
            "ll_hawkes": ll_hawkes,
            "ll_gain": ll_hawkes - ll_base,
            "mu_baseline": scaled.mu,
            "mu_hawkes": hawkes.params[0],
        }
        # Back-compat: keep a single 'mu' column as Hawkes' mu.
        row["mu"] = row["mu_hawkes"]
        for idx, alpha in enumerate(alphas):
            row[f"alpha_type_{idx}"] = alpha
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(results_dir)
    df.to_csv(out_dir / "summary.csv", index=False)
    if not df.empty:
        try:
            from .plots import plot_ll_gain_hist, plot_ll_vs_events, plot_alpha_distribution

            plot_ll_gain_hist(df, out_dir / "ll_gain_hist.png")
            plot_ll_vs_events(df, out_dir / "ll_gain_vs_events.png")
            plot_alpha_distribution(df, out_dir / "alpha_hist.png")
        except Exception:
            pass
    return df
