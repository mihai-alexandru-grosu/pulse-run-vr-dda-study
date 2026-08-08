from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pulse-run-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .constants import (
    OUTCOME_BY_KEY,
    OUTCOME_SPECS,
    PATTERN_PERCENT_COLUMNS,
    STRATEGY_ORDER,
)
from .statistics import welch_anova


STRATEGY_COLORS = ["#4C78A8", "#F58518", "#54A24B"]
PATTERN_COLORS = {
    "Very Easy": "#D8EAF3",
    "Easy": "#8CC2D9",
    "Medium": "#F2C14E",
    "Hard": "#E67E22",
    "Expert": "#B23A48",
}


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, svg_path]


def plot_pattern_composition(
    data: pd.DataFrame,
    output_dir: Path,
    status_label: str,
) -> list[Path]:
    means = []
    group_sizes = []
    for strategy in STRATEGY_ORDER:
        subset = data.loc[data["strategy"].eq(strategy)]
        group_sizes.append(len(subset))
        means.append(
            [
                subset[PATTERN_PERCENT_COLUMNS[label]].mean()
                for label in PATTERN_PERCENT_COLUMNS
            ]
        )
    means_array = np.asarray(means, dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    positions = np.arange(len(STRATEGY_ORDER))
    bottoms = np.zeros(len(STRATEGY_ORDER), dtype=float)
    for category_index, category in enumerate(PATTERN_PERCENT_COLUMNS):
        values = means_array[:, category_index]
        bars = ax.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.68,
            color=PATTERN_COLORS[category],
            edgecolor="white",
            linewidth=0.8,
            label=category,
        )
        for bar, value, bottom in zip(bars, values, bottoms):
            if value >= 7.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bottom + value / 2.0,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#1F1F1F",
                )
        bottoms += values

    labels = [
        f"{strategy}\n(n = {size})"
        for strategy, size in zip(STRATEGY_ORDER, group_sizes)
    ]
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean participant-level pattern proportion (%)")
    ax.set_xlabel("")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        title="Pattern category",
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        frameon=False,
    )
    if status_label != "FINAL":
        ax.set_title(f"{status_label}: not for manuscript use", color="#B23A48")
    fig.tight_layout()
    return _save_figure(fig, output_dir, "figure3_pattern_composition")


def plot_primary_performance_distributions(
    data: pd.DataFrame,
    descriptives: pd.DataFrame,
    output_dir: Path,
    status_label: str,
    random_seed: int,
) -> list[Path]:
    outcomes = [
        ("normalized_target_score_pct", "Normalized target score (%)"),
        ("wall_contact_pct", "Wall-contact percentage"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    rng = np.random.default_rng(random_seed)

    for ax, (outcome, y_label) in zip(axes, outcomes):
        arrays = [
            pd.to_numeric(
                data.loc[data["strategy"].eq(strategy), outcome], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            for strategy in STRATEGY_ORDER
        ]
        box = ax.boxplot(
            arrays,
            positions=np.arange(1, 4),
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.5},
            whiskerprops={"color": "#666666"},
            capprops={"color": "#666666"},
        )
        for patch, color in zip(box["boxes"], STRATEGY_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            patch.set_edgecolor(color)

        for position, (strategy, values, color) in enumerate(
            zip(STRATEGY_ORDER, arrays, STRATEGY_COLORS), start=1
        ):
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(
                position + jitter,
                values,
                s=28,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
                zorder=3,
            )
            row = descriptives.loc[
                descriptives["outcome"].eq(outcome)
                & descriptives["strategy"].eq(strategy)
            ].iloc[0]
            if np.isfinite(row["mean"]):
                ax.errorbar(
                    position,
                    row["mean"],
                    yerr=[
                        [row["mean"] - row["ci_low"]],
                        [row["ci_high"] - row["mean"]],
                    ],
                    fmt="D",
                    markersize=5,
                    color="#111111",
                    ecolor="#111111",
                    capsize=4,
                    linewidth=1.3,
                    zorder=4,
                )
        ax.set_xticks(np.arange(1, 4), STRATEGY_ORDER, rotation=15, ha="right")
        ax.set_ylabel(y_label)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.01,
        "Boxes show the interquartile range; diamonds and whiskers show means and 95% confidence intervals.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    if status_label != "FINAL":
        fig.suptitle(f"{status_label}: not for manuscript use", color="#B23A48")
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    return _save_figure(fig, output_dir, "figure4_performance_distributions")


def plot_likert_distributions(
    data: pd.DataFrame,
    output_dir: Path,
    status_label: str,
) -> list[Path]:
    likert_colors = ["#F1EEF6", "#BDC9E1", "#74A9CF", "#2B8CBE", "#045A8D"]
    outcomes = [
        ("difficulty_rating", "Perceived difficulty"),
        ("enjoyment_rating", "Enjoyment"),
        ("frustration_rating", "Frustration"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.6), sharex=True)
    y_positions = np.arange(len(STRATEGY_ORDER))

    for ax, (outcome, title) in zip(axes, outcomes):
        left = np.zeros(len(STRATEGY_ORDER), dtype=float)
        group_labels = []
        for strategy in STRATEGY_ORDER:
            valid = pd.to_numeric(
                data.loc[data["strategy"].eq(strategy), outcome], errors="coerce"
            ).dropna()
            group_labels.append(f"{strategy} (n = {len(valid)})")
        for rating in range(1, 6):
            percentages = []
            for strategy in STRATEGY_ORDER:
                valid = pd.to_numeric(
                    data.loc[data["strategy"].eq(strategy), outcome], errors="coerce"
                ).dropna()
                percentage = (
                    float(valid.eq(rating).mean() * 100.0) if len(valid) else 0.0
                )
                percentages.append(percentage)
            bars = ax.barh(
                y_positions,
                percentages,
                left=left,
                height=0.58,
                color=likert_colors[rating - 1],
                edgecolor="white",
                linewidth=0.7,
                label=str(rating),
            )
            for bar, percentage, offset in zip(bars, percentages, left):
                if percentage >= 8.0:
                    ax.text(
                        offset + percentage / 2.0,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{percentage:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="white" if rating >= 4 else "#171717",
                    )
            left += np.asarray(percentages)
        ax.set_yticks(y_positions, group_labels)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 100)

    axes[-1].set_xlabel("Participants (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Response",
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
    )
    if status_label != "FINAL":
        fig.suptitle(
            f"{status_label}: not for manuscript use",
            color="#B23A48",
            y=1.07,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(fig, output_dir, "figure5_post_run_ratings")


def diagnostic_outlier_flags(data: pd.DataFrame) -> pd.DataFrame:
    continuous = [spec.key for spec in OUTCOME_SPECS if spec.test == "welch"]
    rows: list[dict[str, object]] = []
    for outcome in continuous:
        for strategy in STRATEGY_ORDER:
            subset = data.loc[data["strategy"].eq(strategy), ["runId", outcome]].copy()
            subset[outcome] = pd.to_numeric(subset[outcome], errors="coerce")
            subset = subset.dropna()
            if subset.empty:
                continue
            values = subset[outcome]
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mean, sd = values.mean(), values.std(ddof=1)
            for _, row in subset.iterrows():
                standardized = (row[outcome] - mean) / sd if sd and np.isfinite(sd) else 0.0
                rows.append(
                    {
                        "outcome": outcome,
                        "strategy": strategy,
                        "run_id": row["runId"],
                        "value": row[outcome],
                        "iqr_low_fence": low_fence,
                        "iqr_high_fence": high_fence,
                        "outside_1_5_iqr": bool(
                            row[outcome] < low_fence or row[outcome] > high_fence
                        ),
                        "within_group_standardized_residual": standardized,
                        "absolute_residual_at_least_3": bool(abs(standardized) >= 3),
                    }
                )
    return pd.DataFrame(rows)


def diagnostic_leave_one_out(data: pd.DataFrame) -> pd.DataFrame:
    continuous = [spec.key for spec in OUTCOME_SPECS if spec.test == "welch"]
    rows: list[dict[str, object]] = []
    for outcome in continuous:
        full_groups = {
            strategy: pd.to_numeric(
                data.loc[data["strategy"].eq(strategy), outcome], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            for strategy in STRATEGY_ORDER
        }
        try:
            full = welch_anova(full_groups)
        except ValueError:
            continue
        eligible = data.loc[data[outcome].notna(), ["runId", "strategy", outcome]]
        for index, row in eligible.iterrows():
            reduced = data.drop(index=index)
            groups = {
                strategy: pd.to_numeric(
                    reduced.loc[reduced["strategy"].eq(strategy), outcome],
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)
                for strategy in STRATEGY_ORDER
            }
            try:
                result = welch_anova(groups)
            except ValueError:
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "removed_run_id": row["runId"],
                    "removed_strategy": row["strategy"],
                    "removed_value": row[outcome],
                    "full_F": full["statistic"],
                    "leave_one_out_F": result["statistic"],
                    "change_in_F": result["statistic"] - full["statistic"],
                    "full_p": full["p_raw"],
                    "leave_one_out_p": result["p_raw"],
                    "change_in_p": result["p_raw"] - full["p_raw"],
                    "full_eta_squared": full["effect_size"],
                    "leave_one_out_eta_squared": result["effect_size"],
                }
            )
    return pd.DataFrame(rows)


def plot_qq_diagnostics(
    data: pd.DataFrame,
    output_dir: Path,
    status_label: str,
) -> list[Path]:
    outcomes = [spec for spec in OUTCOME_SPECS if spec.test == "welch"]
    columns = 3
    rows = math.ceil(len(outcomes) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12.0, 3.6 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for ax, spec in zip(axes_array, outcomes):
        residuals = []
        for strategy in STRATEGY_ORDER:
            values = pd.to_numeric(
                data.loc[data["strategy"].eq(strategy), spec.key], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            if len(values):
                residuals.extend((values - np.mean(values)).tolist())
        if len(residuals) >= 3:
            stats.probplot(np.asarray(residuals), dist="norm", plot=ax)
            ax.get_lines()[0].set_markersize(4)
            ax.get_lines()[0].set_color("#4C78A8")
            ax.get_lines()[1].set_color("#333333")
        ax.set_title(spec.label, fontsize=10)
        ax.set_xlabel("Theoretical quantiles", fontsize=8)
        ax.set_ylabel("Group-centered residuals", fontsize=8)
    for ax in axes_array[len(outcomes) :]:
        ax.axis("off")
    title = "Welch-ANOVA residual Q-Q diagnostics"
    if status_label != "FINAL":
        title += f" ({status_label})"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(fig, output_dir, "diagnostic_qq_residuals")
