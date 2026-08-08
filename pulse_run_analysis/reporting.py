from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import OUTCOME_BY_KEY, STRATEGY_ORDER


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_number(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _descriptive_row(
    descriptives: pd.DataFrame, outcome: str, strategy: str
) -> pd.Series:
    subset = descriptives.loc[
        descriptives["outcome"].eq(outcome)
        & descriptives["strategy"].eq(strategy)
    ]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.iloc[0]


def mean_ci_cell(
    descriptives: pd.DataFrame,
    outcome: str,
    strategy: str,
    digits: int = 1,
) -> str:
    row = _descriptive_row(descriptives, outcome, strategy)
    if row.empty or not np.isfinite(row.get("mean", np.nan)):
        return "NA"
    return (
        f"{row['mean']:.{digits}f} ({row['sd']:.{digits}f}) "
        f"[{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"
    )


def median_iqr_cell(
    descriptives: pd.DataFrame,
    outcome: str,
    strategy: str,
    digits: int = 1,
) -> str:
    row = _descriptive_row(descriptives, outcome, strategy)
    if row.empty or not np.isfinite(row.get("median", np.nan)):
        return "NA"
    return (
        f"{row['median']:.{digits}f} "
        f"[{row['q1']:.{digits}f}, {row['q3']:.{digits}f}]"
    )


def _test_cells(omnibus: pd.DataFrame, outcome: str) -> tuple[str, str, str]:
    row = omnibus.loc[omnibus["outcome"].eq(outcome)]
    if row.empty:
        return "Descriptive", "", ""
    row = row.iloc[0]
    if not np.isfinite(row["statistic"]):
        return "Not estimable", "NA", "NA"
    if row["test"] == "welch":
        statistic = (
            f"Welch F({int(row['df_num'])}, {row['df_den']:.2f}) = "
            f"{row['statistic']:.2f}"
        )
        effect = f"eta-squared = {row['effect_size']:.2f}"
    else:
        statistic = f"H({int(row['df_num'])}) = {row['statistic']:.2f}"
        effect = f"epsilon-squared = {row['effect_size']:.2f}"
    p_value = row["p_holm"] if row["role"] == "primary" else row["p_raw"]
    return statistic, format_p(float(p_value)), effect


def build_table_4(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["Total", *STRATEGY_ORDER]
    rows: list[dict[str, str]] = []

    def subset_for(column: str) -> pd.DataFrame:
        return data if column == "Total" else data.loc[data["strategy"].eq(column)]

    participant_row = {"Characteristic": "Participants, n"}
    age_row = {"Characteristic": "Age, mean (SD)"}
    age_range_row = {"Characteristic": "Age range, years"}
    for column in columns:
        subset = subset_for(column)
        ages = pd.to_numeric(subset["age"], errors="coerce").dropna()
        participant_row[column] = str(len(subset))
        age_row[column] = (
            f"{ages.mean():.1f} ({ages.std(ddof=1):.1f})" if len(ages) > 1 else "NA"
        )
        age_range_row[column] = (
            f"{int(ages.min())}-{int(ages.max())}" if len(ages) else "NA"
        )
    rows.extend([participant_row, age_row, age_range_row])

    for category, label in [
        ("None", "No previous VR experience, n (%)"),
        ("Some", "Some previous VR experience, n (%)"),
        ("High", "High previous VR experience, n (%)"),
    ]:
        row = {"Characteristic": label}
        for column in columns:
            subset = subset_for(column)
            count = int(subset["experience_category"].eq(category).sum())
            percentage = count / len(subset) * 100.0 if len(subset) else np.nan
            row[column] = (
                f"{count} ({percentage:.1f})" if np.isfinite(percentage) else "NA"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _continuous_table(
    row_specs: list[tuple[str, str, int, str]],
    descriptives: pd.DataFrame,
    omnibus: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for outcome, label, digits, mode in row_specs:
        row = {"Outcome": label}
        for strategy in STRATEGY_ORDER:
            if mode in {"median", "descriptive_median"}:
                row[strategy] = median_iqr_cell(
                    descriptives, outcome, strategy, digits
                )
            else:
                row[strategy] = mean_ci_cell(
                    descriptives, outcome, strategy, digits
                )
        if mode in {"descriptive", "descriptive_median"}:
            row["Test statistic"] = "Descriptive"
            row["p-value"] = ""
            row["Effect size"] = ""
        else:
            statistic, p_value, effect = _test_cells(omnibus, outcome)
            row["Test statistic"] = statistic
            row["p-value"] = p_value
            row["Effect size"] = effect
        rows.append(row)
    return pd.DataFrame(rows)


def build_table_5(
    descriptives: pd.DataFrame, omnibus: pd.DataFrame
) -> pd.DataFrame:
    return _continuous_table(
        [
            ("hard_expert_pattern_pct", "Hard/Expert patterns, %", 1, "mean"),
            ("total_patterns_spawned", "Total patterns spawned", 1, "mean"),
            ("resolved_targets", "Resolved targets", 1, "mean"),
            ("maximum_scroll_speed_m_s", "Maximum scroll speed, m/s", 2, "descriptive"),
            ("minimum_pattern_gap_m", "Minimum pattern gap, m", 2, "descriptive"),
        ],
        descriptives,
        omnibus,
    )


def build_table_6(
    descriptives: pd.DataFrame, omnibus: pd.DataFrame
) -> pd.DataFrame:
    return _continuous_table(
        [
            ("normalized_target_score_pct", "Normalized target score, %", 1, "mean"),
            ("wall_contact_pct", "Wall-contact percentage", 1, "mean"),
            ("final_score", "Final score", 1, "mean"),
            ("target_hit_rate_pct", "Target hit rate, %", 1, "mean"),
            ("shot_accuracy_pct", "Shot accuracy, %", 1, "mean"),
            ("color_match_rate_pct", "Color-match rate, %", 1, "mean"),
            ("zero_stability_pct", "Zero-stability percentage", 1, "median"),
        ],
        descriptives,
        omnibus,
    )


def build_table_7(
    data: pd.DataFrame,
    descriptives: pd.DataFrame,
    omnibus: pd.DataFrame,
) -> pd.DataFrame:
    table = _continuous_table(
        [
            (
                "difficulty_rating",
                "Difficulty rating, median [IQR]",
                1,
                "descriptive_median",
            ),
            ("difficulty_deviation", "Difficulty deviation, median [IQR]", 1, "median"),
            ("enjoyment_rating", "Enjoyment, median [IQR]", 1, "median"),
            ("frustration_rating", "Frustration, median [IQR]", 1, "median"),
        ],
        descriptives,
        omnibus,
    )

    motion_row: dict[str, str] = {"Outcome": "Motion sickness reports, n/N (%)"}
    for strategy in STRATEGY_ORDER:
        subset = data.loc[data["strategy"].eq(strategy) & data["feedback_complete"]]
        denominator = len(subset)
        numerator = int(subset["motion_sickness_report"].eq(True).sum())
        percentage = numerator / denominator * 100.0 if denominator else np.nan
        motion_row[strategy] = (
            f"{numerator}/{denominator} ({percentage:.1f})"
            if np.isfinite(percentage)
            else "NA"
        )
    motion_row["Test statistic"] = "Descriptive"
    motion_row["p-value"] = ""
    motion_row["Effect size"] = ""
    return pd.concat([table, pd.DataFrame([motion_row])], ignore_index=True)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(column) for column in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def write_tables_markdown(
    path: Path,
    table_4: pd.DataFrame,
    table_5: pd.DataFrame,
    table_6: pd.DataFrame,
    table_7: pd.DataFrame,
    status_label: str,
) -> None:
    text = f"# Section 4 tables ({status_label})\n\n"
    text += "Table 4. Participant characteristics by difficulty progression strategy.\n\n"
    text += dataframe_to_markdown(table_4) + "\n\n"
    text += "Table 5. Pacing and spawned-pattern outcomes by difficulty progression strategy.\n\n"
    text += dataframe_to_markdown(table_5) + "\n\n"
    text += "Table 6. Gameplay-performance outcomes by difficulty progression strategy.\n\n"
    text += dataframe_to_markdown(table_6) + "\n\n"
    text += "Table 7. Post-run responses by difficulty progression strategy.\n\n"
    text += dataframe_to_markdown(table_7) + "\n\n"
    text += (
        "Continuous cells show mean (SD) [95% CI]. Ordinal and zero-stability "
        "cells show median [IQR]. Holm-adjusted omnibus p-values are shown for "
        "the five primary outcomes. Secondary-outcome p-values are unadjusted "
        "and exploratory. Percentages were calculated at participant level.\n"
    )
    path.write_text(text, encoding="utf-8")


def write_sample_description(
    path: Path,
    record_flow: pd.DataFrame,
    analysis_data: pd.DataFrame,
    status_label: str,
) -> None:
    counts = record_flow.set_index("screening_category")["n"].to_dict()
    retained = int(counts.get("included", 0))
    group_counts = {
        strategy: int(analysis_data["strategy"].eq(strategy).sum())
        for strategy in STRATEGY_ORDER
    }
    feedback_n = int(analysis_data["feedback_complete"].sum())
    raw_n = int(record_flow["n"].sum())
    if status_label == "FINAL":
        lead = "A total of"
        retained_phrase = "The final analysis included"
    else:
        lead = "For the provisional screen, a total of"
        retained_phrase = "The provisional screen currently retains"
    text = (
        f"# Section 4.1 sample text ({status_label})\n\n"
        f"{lead} {raw_n} saved runs were exported from the headsets. Of these, "
        f"{int(counts.get('interrupted_or_incomplete', 0))} were excluded as "
        "interrupted or incomplete sessions, "
        f"{int(counts.get('researcher_staff_demo_or_informal', 0))} as researcher, "
        "staff, demonstration, or informal sessions, "
        f"{int(counts.get('duplicate_or_invalid_record', 0))} as duplicate or "
        "invalid records, "
        f"{int(counts.get('other_ineligible', 0))} for other eligibility reasons, "
        f"and {int(counts.get('eligible_meta_quest_2', 0))} as otherwise eligible "
        "Meta Quest 2 sessions. "
        f"{retained_phrase} {retained} participants: "
        f"{group_counts[STRATEGY_ORDER[0]]} assigned to Fixed Time-Based "
        f"Progression, {group_counts[STRATEGY_ORDER[1]]} to Performance-Based "
        f"DDA, and {group_counts[STRATEGY_ORDER[2]]} to Skill-Specific DDA. "
        "Post-run difficulty, enjoyment, and frustration responses were "
        f"available for {feedback_n} participants. Participant characteristics "
        "by progression strategy are presented in Table 4.\n"
    )
    path.write_text(text, encoding="utf-8")


def write_statistical_sentences(
    path: Path,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    status_label: str,
) -> None:
    lines = [f"# Statistical reporting sentences ({status_label})", ""]
    for _, row in omnibus.iterrows():
        if not np.isfinite(row["statistic"]):
            lines.append(f"{row['label']}: test not estimable ({row['error']}).")
            lines.append("")
            continue
        p_value = row["p_holm"] if row["role"] == "primary" else row["p_raw"]
        p_name = "p_Holm" if row["role"] == "primary" else "p"
        formatted_p = format_p(float(p_value))
        p_expression = (
            f"{p_name} {formatted_p}"
            if formatted_p.startswith("<")
            else f"{p_name} = {formatted_p}"
        )
        finding = (
            "A statistically significant strategy effect was found"
            if row["p_for_decision"] < 0.05
            else "No statistically significant evidence of a strategy difference was found"
        )
        if row["test"] == "welch":
            sentence = (
                f"{row['label']}: {finding}, Welch's F({int(row['df_num'])}, "
                f"{row['df_den']:.2f}) = {row['statistic']:.2f}, "
                f"{p_expression}, eta-squared = "
                f"{row['effect_size']:.2f}."
            )
        else:
            sentence = (
                f"{row['label']}: {finding}, H({int(row['df_num'])}) = "
                f"{row['statistic']:.2f}, {p_expression}, "
                f"epsilon-squared = {row['effect_size']:.2f}."
            )
        lines.extend([sentence, ""])

        outcome_pairs = pairwise.loc[pairwise["outcome"].eq(row["outcome"])]
        significant_pairs = outcome_pairs.loc[
            outcome_pairs["p_pairwise"].lt(0.05)
        ]
        if not outcome_pairs.empty and significant_pairs.empty:
            method = str(outcome_pairs.iloc[0]["method"])
            lines.extend(
                [
                    f"No {method} pairwise comparison reached statistical significance.",
                    "",
                ]
            )
        for _, pair in significant_pairs.iterrows():
            if pair["method"] == "Games-Howell":
                pair_p = format_p(float(pair["p_pairwise"]))
                pair_p_expression = (
                    f"p {pair_p}" if pair_p.startswith("<") else f"p = {pair_p}"
                )
                lines.append(
                    f"Games-Howell: {pair['group_1']} minus {pair['group_2']}, "
                    f"mean difference = {pair['mean_difference']:.2f}, 95% CI "
                    f"[{pair['difference_ci_low']:.2f}, {pair['difference_ci_high']:.2f}], "
                    f"{pair_p_expression}, Hedges' g = "
                    f"{pair['effect_size']:.2f}, 95% CI [{pair['effect_ci_low']:.2f}, "
                    f"{pair['effect_ci_high']:.2f}]."
                )
            else:
                pair_p = format_p(float(pair["p_pairwise"]))
                pair_p_expression = (
                    f"Holm-adjusted p {pair_p}"
                    if pair_p.startswith("<")
                    else f"Holm-adjusted p = {pair_p}"
                )
                lines.append(
                    f"Dunn-Holm: {pair['group_1']} versus {pair['group_2']}, "
                    f"{pair_p_expression}, "
                    f"Cliff's delta = {pair['effect_size']:.2f}, 95% CI "
                    f"[{pair['effect_ci_low']:.2f}, {pair['effect_ci_high']:.2f}]."
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
