from __future__ import annotations

import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    OUTCOME_SPECS,
    PATTERN_PERCENT_COLUMNS,
    PRIMARY_OUTCOMES,
    STRATEGY_ORDER,
)
from .data import (
    AnalysisConfig,
    attach_review_metadata,
    derive_outcomes,
    initialize_review_files,
    load_json_records,
    screen_records,
    validate_records,
)
from .plots import (
    diagnostic_leave_one_out,
    diagnostic_outlier_flags,
    plot_likert_distributions,
    plot_pattern_composition,
    plot_primary_performance_distributions,
    plot_qq_diagnostics,
)
from .reporting import (
    build_table_4,
    build_table_5,
    build_table_6,
    build_table_7,
    write_sample_description,
    write_statistical_sentences,
    write_tables_markdown,
)
from .statistics import group_descriptives, run_omnibus_tests


@dataclass
class AnalysisResult:
    status: dict[str, object]
    record_flow: pd.DataFrame
    validation_report: pd.DataFrame
    screening_log: pd.DataFrame
    analysis_data: pd.DataFrame
    descriptives: pd.DataFrame
    omnibus_tests: pd.DataFrame
    pairwise_tests: pd.DataFrame
    table_4: pd.DataFrame
    table_5: pd.DataFrame
    table_6: pd.DataFrame
    table_7: pd.DataFrame
    figure_paths: list[Path]
    output_dir: Path


@dataclass
class SensitivityResult:
    status: dict[str, object]
    record_flow: pd.DataFrame
    screening_log: pd.DataFrame
    analysis_data: pd.DataFrame
    descriptives: pd.DataFrame
    omnibus_tests: pd.DataFrame
    pairwise_tests: pd.DataFrame
    headset_distribution: pd.DataFrame
    comparison_with_primary: pd.DataFrame
    output_dir: Path


@dataclass
class CompleteAnalysisResult:
    primary: AnalysisResult
    sensitivity: SensitivityResult


SENSITIVITY_HEADSETS = ("Meta Quest 3", "Meta Quest 2")


def refresh_review_templates(config_path: str | Path) -> tuple[Path, Path]:
    config = AnalysisConfig.from_toml(config_path)
    raw, files = load_json_records(config)
    initialize_review_files(raw, files, config)
    return config.source_manifest, config.session_review


def _software_versions() -> pd.DataFrame:
    packages = ["numpy", "pandas", "scipy", "matplotlib"]
    rows = [{"software": "Python", "version": platform.python_version()}]
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        rows.append({"software": package, "version": version})
    return pd.DataFrame(rows)


def _screening_log(screened: pd.DataFrame) -> pd.DataFrame:
    desired = [
        "record_key",
        "source_file",
        "source_row",
        "runId",
        "participantId",
        "startedUtc",
        "attemptIndex",
        "ddaAlgorithm",
        "age",
        "experienceLabel",
        "movementAnalytics.sampledDurationSeconds",
        "actualRunDurationSeconds",
        "tutorialAnalytics.completed",
        "postRunFeedback.completed",
        "headset_model",
        "session_type",
        "attempt_decision",
        "manual_exclusion_reason",
        "screening_status",
        "exclusion_reason",
        "notes",
    ]
    available = [column for column in desired if column in screened]
    return screened[available].copy()


def _status_summary(
    raw: pd.DataFrame,
    analysis_data: pd.DataFrame,
    unresolved_ids: list[str],
    validation_report: pd.DataFrame,
    omnibus: pd.DataFrame,
    config: AnalysisConfig,
) -> dict[str, object]:
    group_sizes = {
        strategy: int(analysis_data["strategy"].eq(strategy).sum())
        for strategy in STRATEGY_ORDER
    }
    feedback_n = int(analysis_data["feedback_complete"].sum())
    nonestimable_primary = omnibus.loc[
        omnibus["role"].eq("primary") & omnibus["p_raw"].isna(), "outcome"
    ].astype(str).tolist()
    status_label = (
        "FINAL"
        if (
            config.strict_manual_review
            and not unresolved_ids
            and not nonestimable_primary
        )
        else "PROVISIONAL"
    )
    return {
        "analysis_status": status_label,
        "raw_records": int(len(raw)),
        "analyzed_sessions": int(len(analysis_data)),
        "group_sizes": group_sizes,
        "complete_post_run_responses": feedback_n,
        "missing_complete_post_run_responses": int(len(analysis_data) - feedback_n),
        "unresolved_review_records": len(unresolved_ids),
        "unresolved_run_ids": unresolved_ids,
        "validation_errors": int(validation_report["severity"].eq("error").sum()),
        "validation_warnings": int(
            validation_report["severity"].eq("warning").sum()
        ),
        "nonestimable_primary_tests": nonestimable_primary,
        "active_duration_minimum_seconds": config.active_duration_min_s,
        "target_headset": config.target_headset,
        "primary_outcomes": [
            spec.key for spec in OUTCOME_SPECS if spec.role == "primary"
        ],
        "secondary_outcomes": [
            spec.key for spec in OUTCOME_SPECS if spec.role == "secondary"
        ],
    }


def run_analysis(config_path: str | Path) -> AnalysisResult:
    config = AnalysisConfig.from_toml(config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = config.output_dir / "tables"
    figure_dir = config.output_dir / "figures"
    diagnostic_dir = config.output_dir / "diagnostics"
    for directory in [table_dir, figure_dir, diagnostic_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    raw, files = load_json_records(config)
    initialize_review_files(raw, files, config)
    reviewed = attach_review_metadata(raw, config)
    validation_report, checked = validate_records(reviewed, config)
    screened, record_flow, unresolved_ids = screen_records(checked, config)
    analysis_data = derive_outcomes(screened)

    descriptive_outcomes = [spec.key for spec in OUTCOME_SPECS]
    descriptive_outcomes.extend(
        [
            "maximum_scroll_speed_m_s",
            "minimum_pattern_gap_m",
            "difficulty_rating",
            *PATTERN_PERCENT_COLUMNS.values(),
        ]
    )
    descriptive_outcomes = list(dict.fromkeys(descriptive_outcomes))
    descriptives = group_descriptives(
        analysis_data, descriptive_outcomes, config.ci_level
    )
    omnibus, pairwise = run_omnibus_tests(
        analysis_data,
        config.alpha,
        config.ci_level,
        config.bootstrap_resamples,
        config.random_seed,
    )
    nonestimable_primary = omnibus.loc[
        omnibus["role"].eq("primary") & omnibus["p_raw"].isna(), "outcome"
    ].astype(str).tolist()
    if config.strict_manual_review and nonestimable_primary:
        raise RuntimeError(
            "Final-mode analysis is blocked because these primary tests are "
            "not estimable: " + ", ".join(nonestimable_primary)
        )

    table_4 = build_table_4(analysis_data)
    table_5 = build_table_5(descriptives, omnibus)
    table_6 = build_table_6(descriptives, omnibus)
    table_7 = build_table_7(analysis_data, descriptives, omnibus)

    status = _status_summary(
        raw,
        analysis_data,
        unresolved_ids,
        validation_report,
        omnibus,
        config,
    )
    status_label = str(status["analysis_status"])

    screening_log = _screening_log(screened)
    analysis_data.to_csv(config.output_dir / "analysis_dataset.csv", index=False)
    screening_log.to_csv(config.output_dir / "screening_log.csv", index=False)
    record_flow.to_csv(config.output_dir / "record_flow.csv", index=False)
    validation_report.to_csv(
        config.output_dir / "validation_report.csv", index=False
    )
    descriptives.to_csv(
        config.output_dir / "group_descriptives.csv", index=False
    )
    omnibus.to_csv(config.output_dir / "omnibus_tests.csv", index=False)
    pairwise.to_csv(config.output_dir / "pairwise_tests.csv", index=False)
    _software_versions().to_csv(
        config.output_dir / "software_versions.csv", index=False
    )

    for number, table in [
        (4, table_4),
        (5, table_5),
        (6, table_6),
        (7, table_7),
    ]:
        table.to_csv(table_dir / f"table_{number}.csv", index=False)

    write_tables_markdown(
        table_dir / "section4_tables.md",
        table_4,
        table_5,
        table_6,
        table_7,
        status_label,
    )
    write_sample_description(
        table_dir / "section4_1_sample_text.md",
        record_flow,
        analysis_data,
        status_label,
    )
    write_statistical_sentences(
        table_dir / "statistical_sentences.md",
        omnibus,
        pairwise,
        status_label,
    )

    figure_paths: list[Path] = []
    figure_paths.extend(
        plot_pattern_composition(analysis_data, figure_dir, status_label)
    )
    figure_paths.extend(
        plot_primary_performance_distributions(
            analysis_data,
            descriptives,
            figure_dir,
            status_label,
            config.random_seed,
        )
    )
    figure_paths.extend(
        plot_likert_distributions(analysis_data, figure_dir, status_label)
    )
    figure_paths.extend(
        plot_qq_diagnostics(analysis_data, diagnostic_dir, status_label)
    )

    outlier_flags = diagnostic_outlier_flags(analysis_data)
    leave_one_out = diagnostic_leave_one_out(analysis_data)
    outlier_flags.to_csv(diagnostic_dir / "outlier_flags.csv", index=False)
    leave_one_out.to_csv(
        diagnostic_dir / "leave_one_out_influence.csv", index=False
    )

    status_path = config.output_dir / "analysis_status.json"
    status_path.write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return AnalysisResult(
        status=status,
        record_flow=record_flow,
        validation_report=validation_report,
        screening_log=screening_log,
        analysis_data=analysis_data,
        descriptives=descriptives,
        omnibus_tests=omnibus,
        pairwise_tests=pairwise,
        table_4=table_4,
        table_5=table_5,
        table_6=table_6,
        table_7=table_7,
        figure_paths=figure_paths,
        output_dir=config.output_dir,
    )


def _headset_distribution(data: pd.DataFrame) -> pd.DataFrame:
    distribution = pd.crosstab(
        data["headset_model"],
        data["strategy"],
        dropna=False,
    ).reindex(columns=STRATEGY_ORDER, fill_value=0)
    preferred_order = [
        headset for headset in SENSITIVITY_HEADSETS if headset in distribution.index
    ]
    remaining = [
        headset for headset in distribution.index if headset not in preferred_order
    ]
    distribution = distribution.reindex([*preferred_order, *remaining])
    distribution["Total"] = distribution.sum(axis=1)
    total_row = distribution.sum(axis=0).to_frame().T
    total_row.index = ["Total"]
    distribution = pd.concat([distribution, total_row])
    return distribution.reset_index(names="Headset")


def _compare_primary_and_sensitivity(
    primary: AnalysisResult,
    sensitivity_omnibus: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "outcome",
        "label",
        "test",
        "n_total",
        "statistic",
        "df_num",
        "df_den",
        "p_raw",
        "p_holm",
        "effect_size",
        "effect_size_name",
        "significant",
    ]
    primary_tests = primary.omnibus_tests.loc[
        primary.omnibus_tests["role"].eq("primary"), columns
    ].copy()
    sensitivity_tests = sensitivity_omnibus[columns].copy()
    comparison = primary_tests.merge(
        sensitivity_tests,
        on=["outcome", "label", "test"],
        how="outer",
        suffixes=("_quest3_primary", "_all_headsets"),
        validate="one_to_one",
    )
    comparison["significance_changed"] = (
        comparison["significant_quest3_primary"]
        != comparison["significant_all_headsets"]
    )
    outcome_order = {key: index for index, key in enumerate(PRIMARY_OUTCOMES)}
    comparison["_order"] = comparison["outcome"].map(outcome_order)
    return comparison.sort_values("_order").drop(columns="_order").reset_index(
        drop=True
    )


def run_sensitivity_analysis(
    config_path: str | Path,
    primary_result: AnalysisResult,
) -> SensitivityResult:
    """Run the prespecified all-eligible-headset sensitivity analysis.

    This pass retains eligible Meta Quest 2 and Meta Quest 3 participant sessions,
    tests only the five primary outcomes, and writes to a separate subdirectory.
    Primary-analysis files are never overwritten.
    """
    config = AnalysisConfig.from_toml(config_path)
    output_dir = config.output_dir / "sensitivity_all_eligible_headsets"
    table_dir = output_dir / "tables"
    for directory in [output_dir, table_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    raw, files = load_json_records(config)
    initialize_review_files(raw, files, config)
    reviewed = attach_review_metadata(raw, config)
    validation_report, checked = validate_records(reviewed, config)
    screened, record_flow, unresolved_ids = screen_records(
        checked,
        config,
        included_headsets=SENSITIVITY_HEADSETS,
    )
    analysis_data = derive_outcomes(screened)

    descriptives = group_descriptives(
        analysis_data,
        PRIMARY_OUTCOMES,
        config.ci_level,
    )
    omnibus_all, pairwise_all = run_omnibus_tests(
        analysis_data,
        config.alpha,
        config.ci_level,
        config.bootstrap_resamples,
        config.random_seed,
    )
    omnibus = omnibus_all.loc[omnibus_all["role"].eq("primary")].copy()
    pairwise = pairwise_all.loc[
        pairwise_all["outcome"].isin(PRIMARY_OUTCOMES)
    ].copy()

    nonestimable_primary = omnibus.loc[
        omnibus["p_raw"].isna(), "outcome"
    ].astype(str).tolist()
    if config.strict_manual_review and nonestimable_primary:
        raise RuntimeError(
            "Final-mode sensitivity analysis is blocked because these primary "
            "tests are not estimable: " + ", ".join(nonestimable_primary)
        )

    status_label = (
        "FINAL"
        if (
            config.strict_manual_review
            and not unresolved_ids
            and not nonestimable_primary
        )
        else "PROVISIONAL"
    )
    group_sizes = {
        strategy: int(analysis_data["strategy"].eq(strategy).sum())
        for strategy in STRATEGY_ORDER
    }
    headset_counts = {
        str(headset): int(count)
        for headset, count in analysis_data["headset_model"].value_counts().items()
    }
    status = {
        "analysis_status": status_label,
        "analysis_type": "sensitivity_all_eligible_headsets",
        "raw_records": int(len(raw)),
        "analyzed_sessions": int(len(analysis_data)),
        "group_sizes": group_sizes,
        "included_headsets": list(SENSITIVITY_HEADSETS),
        "headset_counts": headset_counts,
        "complete_post_run_responses": int(analysis_data["feedback_complete"].sum()),
        "unresolved_review_records": len(unresolved_ids),
        "unresolved_run_ids": unresolved_ids,
        "nonestimable_primary_tests": nonestimable_primary,
        "primary_outcomes": PRIMARY_OUTCOMES,
    }

    sensitivity_flow = record_flow.copy()
    sensitivity_flow.loc[
        sensitivity_flow["screening_category"].eq("included"), "description"
    ] = "Included in all-eligible-headset sensitivity analysis"
    screening_log = _screening_log(screened)
    headset_distribution = _headset_distribution(analysis_data)
    comparison = _compare_primary_and_sensitivity(primary_result, omnibus)

    analysis_data.to_csv(output_dir / "analysis_dataset.csv", index=False)
    screening_log.to_csv(output_dir / "screening_log.csv", index=False)
    sensitivity_flow.to_csv(output_dir / "record_flow.csv", index=False)
    validation_report.to_csv(output_dir / "validation_report.csv", index=False)
    descriptives.to_csv(
        output_dir / "group_descriptives_primary.csv", index=False
    )
    omnibus.to_csv(output_dir / "omnibus_tests_primary.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_tests_primary.csv", index=False)
    headset_distribution.to_csv(
        output_dir / "headset_distribution.csv", index=False
    )
    comparison.to_csv(
        output_dir / "comparison_with_quest3_primary.csv", index=False
    )
    write_statistical_sentences(
        table_dir / "statistical_sentences_primary.md",
        omnibus,
        pairwise,
        status_label,
    )
    (output_dir / "analysis_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return SensitivityResult(
        status=status,
        record_flow=sensitivity_flow,
        screening_log=screening_log,
        analysis_data=analysis_data,
        descriptives=descriptives,
        omnibus_tests=omnibus,
        pairwise_tests=pairwise,
        headset_distribution=headset_distribution,
        comparison_with_primary=comparison,
        output_dir=output_dir,
    )


def run_complete_analysis(config_path: str | Path) -> CompleteAnalysisResult:
    """Run the Quest 3 primary analysis followed by the all-headset check."""
    primary = run_analysis(config_path)
    sensitivity = run_sensitivity_analysis(config_path, primary)
    return CompleteAnalysisResult(primary=primary, sensitivity=sensitivity)
