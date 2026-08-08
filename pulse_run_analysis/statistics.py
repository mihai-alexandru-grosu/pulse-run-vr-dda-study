from __future__ import annotations

import math
import zlib
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .constants import OUTCOME_SPECS, STRATEGY_ORDER


def _outcome_groups(data: pd.DataFrame, outcome: str) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    for strategy in STRATEGY_ORDER:
        values = pd.to_numeric(
            data.loc[data["strategy"].eq(strategy), outcome], errors="coerce"
        ).dropna()
        groups[strategy] = values.to_numpy(dtype=float)
    return groups


def eta_squared(groups: dict[str, np.ndarray]) -> float:
    arrays = [values for values in groups.values() if len(values)]
    all_values = np.concatenate(arrays)
    if len(all_values) == 0:
        return np.nan
    grand_mean = float(np.mean(all_values))
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    if ss_total == 0:
        return 0.0
    ss_between = sum(
        len(values) * (float(np.mean(values)) - grand_mean) ** 2
        for values in arrays
    )
    return float(ss_between / ss_total)


def welch_anova(groups: dict[str, np.ndarray]) -> dict[str, float]:
    arrays = list(groups.values())
    k = len(arrays)
    if k < 2 or any(len(values) < 2 for values in arrays):
        raise ValueError("Welch ANOVA requires at least two observations per group")

    sizes = np.array([len(values) for values in arrays], dtype=float)
    means = np.array([np.mean(values) for values in arrays], dtype=float)
    variances = np.array([np.var(values, ddof=1) for values in arrays], dtype=float)
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        raise ValueError("Welch ANOVA is not estimable when a group variance is zero")

    weights = sizes / variances
    weight_sum = float(np.sum(weights))
    weighted_mean = float(np.sum(weights * means) / weight_sum)
    numerator = float(
        np.sum(weights * (means - weighted_mean) ** 2) / (k - 1)
    )
    correction_term = float(
        np.sum(((1.0 - weights / weight_sum) ** 2) / (sizes - 1.0))
    )
    adjustment = 1.0 + (2.0 * (k - 2.0) / (k**2 - 1.0)) * correction_term
    statistic = numerator / adjustment
    df_num = float(k - 1)
    df_den = float((k**2 - 1.0) / (3.0 * correction_term))
    p_value = float(stats.f.sf(statistic, df_num, df_den))
    return {
        "statistic": statistic,
        "df_num": df_num,
        "df_den": df_den,
        "p_raw": p_value,
        "effect_size": eta_squared(groups),
        "effect_size_name": "eta_squared",
    }


def kruskal_wallis(groups: dict[str, np.ndarray]) -> dict[str, float]:
    arrays = list(groups.values())
    if any(len(values) == 0 for values in arrays):
        raise ValueError("Kruskal-Wallis requires at least one observation per group")
    all_values = np.concatenate(arrays)
    if np.all(all_values == all_values[0]):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = stats.kruskal(*arrays, nan_policy="omit")
        statistic, p_value = float(statistic), float(p_value)
    total_n = len(all_values)
    k = len(arrays)
    epsilon = (statistic - k + 1.0) / (total_n - k) if total_n > k else np.nan
    if np.isfinite(epsilon):
        epsilon = max(0.0, float(epsilon))
    return {
        "statistic": statistic,
        "df_num": float(k - 1),
        "df_den": np.nan,
        "p_raw": p_value,
        "effect_size": epsilon,
        "effect_size_name": "epsilon_squared",
    }


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if len(valid_positions) == 0:
        return adjusted

    valid_values = values[valid_positions]
    order = np.argsort(valid_values)
    ordered = valid_values[order]
    m = len(ordered)
    ordered_adjusted = np.maximum.accumulate(
        np.array([(m - rank) * p for rank, p in enumerate(ordered)])
    )
    ordered_adjusted = np.minimum(ordered_adjusted, 1.0)
    restored = np.empty(m, dtype=float)
    restored[order] = ordered_adjusted
    adjusted[valid_positions] = restored
    return adjusted


def hedges_g(first: np.ndarray, second: np.ndarray) -> float:
    n1, n2 = len(first), len(second)
    if n1 < 2 or n2 < 2:
        return np.nan
    variance_1 = float(np.var(first, ddof=1))
    variance_2 = float(np.var(second, ddof=1))
    pooled_variance = (
        (n1 - 1) * variance_1 + (n2 - 1) * variance_2
    ) / (n1 + n2 - 2)
    if pooled_variance <= 0:
        return 0.0 if np.mean(first) == np.mean(second) else np.nan
    cohen_d = (float(np.mean(first)) - float(np.mean(second))) / math.sqrt(
        pooled_variance
    )
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(correction * cohen_d)


def cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) == 0 or len(second) == 0:
        return np.nan
    differences = first[:, None] - second[None, :]
    return float(
        (np.count_nonzero(differences > 0) - np.count_nonzero(differences < 0))
        / differences.size
    )


def bootstrap_two_sample_ci(
    first: np.ndarray,
    second: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    confidence_level: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample_first = rng.choice(first, size=len(first), replace=True)
        sample_second = rng.choice(second, size=len(second), replace=True)
        values[index] = statistic(sample_first, sample_second)
    values = values[np.isfinite(values)]
    if len(values) < max(100, int(resamples * 0.8)):
        return np.nan, np.nan
    alpha = 1.0 - confidence_level
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def games_howell(
    groups: dict[str, np.ndarray],
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    k = len(groups)
    alpha = 1.0 - confidence_level
    for pair_index, (name_1, name_2) in enumerate(combinations(groups, 2)):
        first, second = groups[name_1], groups[name_2]
        n1, n2 = len(first), len(second)
        variance_1 = float(np.var(first, ddof=1))
        variance_2 = float(np.var(second, ddof=1))
        component_1 = variance_1 / n1
        component_2 = variance_2 / n2
        se = math.sqrt(component_1 + component_2)
        difference = float(np.mean(first) - np.mean(second))
        if se == 0:
            degrees_freedom = np.inf
            t_value = np.inf if difference else 0.0
        else:
            degrees_freedom = (component_1 + component_2) ** 2 / (
                component_1**2 / (n1 - 1) + component_2**2 / (n2 - 1)
            )
            t_value = difference / se
        q_value = math.sqrt(2.0) * abs(t_value)
        p_value = float(stats.studentized_range.sf(q_value, k, degrees_freedom))
        q_critical = float(
            stats.studentized_range.ppf(
                1.0 - alpha, k, degrees_freedom
            )
        )
        half_width = q_critical * se / math.sqrt(2.0)
        effect = hedges_g(first, second)
        pair_seed = seed + pair_index * 7919
        effect_low, effect_high = bootstrap_two_sample_ci(
            first,
            second,
            hedges_g,
            confidence_level,
            bootstrap_resamples,
            pair_seed,
        )
        rows.append(
            {
                "group_1": name_1,
                "group_2": name_2,
                "n_1": n1,
                "n_2": n2,
                "mean_difference": difference,
                "difference_ci_low": difference - half_width,
                "difference_ci_high": difference + half_width,
                "pairwise_statistic": t_value,
                "pairwise_df": degrees_freedom,
                "p_pairwise": p_value,
                "effect_size_name": "hedges_g",
                "effect_size": effect,
                "effect_ci_low": effect_low,
                "effect_ci_high": effect_high,
            }
        )
    return pd.DataFrame(rows)


def dunn_test(
    groups: dict[str, np.ndarray],
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
) -> pd.DataFrame:
    labels: list[str] = []
    values: list[float] = []
    for name, group_values in groups.items():
        labels.extend([name] * len(group_values))
        values.extend(group_values.tolist())

    frame = pd.DataFrame({"group": labels, "value": values})
    frame["rank"] = stats.rankdata(frame["value"], method="average")
    total_n = len(frame)
    tie_counts = frame["value"].value_counts().to_numpy(dtype=float)
    tie_sum = float(np.sum(tie_counts**3 - tie_counts))
    variance_term = total_n * (total_n + 1.0) / 12.0
    if total_n > 1:
        variance_term -= tie_sum / (12.0 * (total_n - 1.0))

    rows: list[dict[str, float | str | int]] = []
    for pair_index, (name_1, name_2) in enumerate(combinations(groups, 2)):
        first, second = groups[name_1], groups[name_2]
        mean_rank_1 = float(frame.loc[frame["group"].eq(name_1), "rank"].mean())
        mean_rank_2 = float(frame.loc[frame["group"].eq(name_2), "rank"].mean())
        denominator = math.sqrt(
            variance_term * (1.0 / len(first) + 1.0 / len(second))
        )
        z_value = (
            (mean_rank_1 - mean_rank_2) / denominator if denominator else 0.0
        )
        p_value = float(2.0 * stats.norm.sf(abs(z_value)))
        effect = cliffs_delta(first, second)
        pair_seed = seed + pair_index * 7919
        effect_low, effect_high = bootstrap_two_sample_ci(
            first,
            second,
            cliffs_delta,
            confidence_level,
            bootstrap_resamples,
            pair_seed,
        )
        rows.append(
            {
                "group_1": name_1,
                "group_2": name_2,
                "n_1": len(first),
                "n_2": len(second),
                "median_difference": float(np.median(first) - np.median(second)),
                "pairwise_statistic": z_value,
                "pairwise_df": np.nan,
                "p_pairwise_raw": p_value,
                "effect_size_name": "cliffs_delta",
                "effect_size": effect,
                "effect_ci_low": effect_low,
                "effect_ci_high": effect_high,
            }
        )
    result = pd.DataFrame(rows)
    result["p_pairwise"] = holm_adjust(result["p_pairwise_raw"].to_numpy())
    return result


def group_descriptives(
    data: pd.DataFrame,
    outcomes: list[str],
    confidence_level: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    alpha = 1.0 - confidence_level
    for outcome in outcomes:
        for strategy in STRATEGY_ORDER:
            values = pd.to_numeric(
                data.loc[data["strategy"].eq(strategy), outcome], errors="coerce"
            ).dropna()
            array = values.to_numpy(dtype=float)
            n = len(array)
            mean = float(np.mean(array)) if n else np.nan
            sd = float(np.std(array, ddof=1)) if n > 1 else np.nan
            if n > 1:
                half_width = float(
                    stats.t.ppf(1.0 - alpha / 2.0, n - 1) * sd / math.sqrt(n)
                )
                ci_low, ci_high = mean - half_width, mean + half_width
            else:
                ci_low, ci_high = np.nan, np.nan
            rows.append(
                {
                    "outcome": outcome,
                    "strategy": strategy,
                    "n": n,
                    "mean": mean,
                    "sd": sd,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "median": float(np.median(array)) if n else np.nan,
                    "q1": float(np.quantile(array, 0.25)) if n else np.nan,
                    "q3": float(np.quantile(array, 0.75)) if n else np.nan,
                    "minimum": float(np.min(array)) if n else np.nan,
                    "maximum": float(np.max(array)) if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_omnibus_tests(
    data: pd.DataFrame,
    alpha: float,
    confidence_level: float,
    bootstrap_resamples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str | int]] = []
    group_cache: dict[str, dict[str, np.ndarray]] = {}
    for spec in OUTCOME_SPECS:
        groups = _outcome_groups(data, spec.key)
        group_cache[spec.key] = groups
        try:
            result = (
                welch_anova(groups)
                if spec.test == "welch"
                else kruskal_wallis(groups)
            )
            error = ""
        except ValueError as exc:
            result = {
                "statistic": np.nan,
                "df_num": np.nan,
                "df_den": np.nan,
                "p_raw": np.nan,
                "effect_size": np.nan,
                "effect_size_name": (
                    "eta_squared" if spec.test == "welch" else "epsilon_squared"
                ),
            }
            error = str(exc)
        rows.append(
            {
                "outcome": spec.key,
                "label": spec.label,
                "section": spec.section,
                "role": spec.role,
                "test": spec.test,
                "n_total": sum(len(values) for values in groups.values()),
                "n_fixed": len(groups[STRATEGY_ORDER[0]]),
                "n_performance": len(groups[STRATEGY_ORDER[1]]),
                "n_skill_specific": len(groups[STRATEGY_ORDER[2]]),
                **result,
                "error": error,
            }
        )

    omnibus = pd.DataFrame(rows)
    omnibus["p_holm"] = np.nan
    primary_mask = omnibus["role"].eq("primary")
    primary_p_values = omnibus.loc[primary_mask, "p_raw"].to_numpy()
    if np.isfinite(primary_p_values).all():
        omnibus.loc[primary_mask, "p_holm"] = holm_adjust(primary_p_values)
    omnibus["p_for_decision"] = np.where(
        primary_mask, omnibus["p_holm"], omnibus["p_raw"]
    )
    omnibus["significant"] = omnibus["p_for_decision"].lt(alpha)

    pairwise_frames: list[pd.DataFrame] = []
    for _, omnibus_row in omnibus.loc[omnibus["significant"]].iterrows():
        outcome = str(omnibus_row["outcome"])
        groups = group_cache[outcome]
        stable_seed = random_seed + zlib.crc32(outcome.encode("utf-8"))
        if omnibus_row["test"] == "welch":
            pairwise = games_howell(
                groups,
                confidence_level,
                bootstrap_resamples,
                stable_seed,
            )
            pairwise["method"] = "Games-Howell"
        else:
            pairwise = dunn_test(
                groups,
                confidence_level,
                bootstrap_resamples,
                stable_seed,
            )
            pairwise["method"] = "Dunn-Holm"
        pairwise.insert(0, "outcome", outcome)
        pairwise.insert(1, "label", omnibus_row["label"])
        pairwise_frames.append(pairwise)

    if pairwise_frames:
        pairwise_results = pd.concat(pairwise_frames, ignore_index=True, sort=False)
    else:
        pairwise_results = pd.DataFrame(
            columns=[
                "outcome",
                "label",
                "group_1",
                "group_2",
                "method",
                "p_pairwise",
                "effect_size_name",
                "effect_size",
                "effect_ci_low",
                "effect_ci_high",
            ]
        )
    return omnibus, pairwise_results
