from __future__ import annotations

import glob
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    ALLOWED_ATTEMPT_DECISIONS,
    ALLOWED_MANUAL_REASONS,
    ALLOWED_SESSION_TYPES,
    EXPERIENCE_MAP,
    FLOW_LABELS,
    FLOW_ORDER,
    NONPARTICIPANT_SESSION_TYPES,
    PATTERN_COLUMNS,
    STRATEGY_MAP,
)


class ReviewRequiredError(RuntimeError):
    """Raised when final-mode analysis still has unresolved record metadata."""


@dataclass(frozen=True)
class AnalysisConfig:
    config_path: Path
    data_globs: tuple[str, ...]
    output_dir: Path
    source_manifest: Path
    session_review: Path
    target_headset: str
    active_duration_min_s: float
    strict_manual_review: bool
    expected_age_min: int | None
    expected_age_max: int | None
    alpha: float
    ci_level: float
    bootstrap_resamples: int
    random_seed: int

    @classmethod
    def from_toml(cls, path: str | Path) -> "AnalysisConfig":
        config_path = Path(path).resolve()
        base = config_path.parent
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

        paths = raw.get("paths", {})
        screening = raw.get("screening", {})
        analysis = raw.get("analysis", {})

        patterns = tuple(paths.get("data_globs", ["data/raw/*.json"]))
        if not patterns:
            raise ValueError("paths.data_globs must contain at least one pattern")

        def resolved(value: str) -> Path:
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

        return cls(
            config_path=config_path,
            data_globs=patterns,
            output_dir=resolved(paths.get("output_dir", "outputs")),
            source_manifest=resolved(
                paths.get("source_manifest", "review/source_manifest.csv")
            ),
            session_review=resolved(
                paths.get("session_review", "review/session_review.csv")
            ),
            target_headset=str(screening.get("target_headset", "Meta Quest 3")),
            active_duration_min_s=float(
                screening.get("active_duration_min_s", 209.0)
            ),
            strict_manual_review=bool(
                screening.get("strict_manual_review", False)
            ),
            expected_age_min=_optional_int(screening.get("expected_age_min")),
            expected_age_max=_optional_int(screening.get("expected_age_max")),
            alpha=float(analysis.get("alpha", 0.05)),
            ci_level=float(analysis.get("ci_level", 0.95)),
            bootstrap_resamples=int(analysis.get("bootstrap_resamples", 5000)),
            random_seed=int(analysis.get("random_seed", 20260804)),
        )

    @property
    def base_dir(self) -> Path:
        return self.config_path.parent


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def discover_json_files(config: AnalysisConfig) -> list[Path]:
    files: list[Path] = []
    for pattern in config.data_globs:
        expanded = pattern if Path(pattern).is_absolute() else str(config.base_dir / pattern)
        files.extend(Path(item).resolve() for item in glob.glob(expanded))
    unique = sorted(set(files))
    if not unique:
        joined = ", ".join(config.data_globs)
        raise FileNotFoundError(f"No JSON files matched: {joined}")
    return unique


def load_json_records(config: AnalysisConfig) -> tuple[pd.DataFrame, list[Path]]:
    files = discover_json_files(config)
    frames: list[pd.DataFrame] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
            records = payload["runs"]
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(
                f"{path.name} must contain a top-level runs array or a JSON array"
            )

        frame = pd.json_normalize(records, sep=".")
        frame["source_file"] = path.name
        frame["source_row"] = np.arange(1, len(frame) + 1)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["record_key"] = (
        data["source_file"].astype(str)
        + ":"
        + data["source_row"].astype(str)
    )
    return data, files


def initialize_review_files(
    raw: pd.DataFrame,
    files: list[Path],
    config: AnalysisConfig,
) -> None:
    """Create review CSVs if absent and append rows for newly observed records."""
    config.source_manifest.parent.mkdir(parents=True, exist_ok=True)
    config.session_review.parent.mkdir(parents=True, exist_ok=True)

    manifest_columns = ["source_file", "headset_model", "collection_round", "notes"]
    if config.source_manifest.exists():
        manifest = pd.read_csv(config.source_manifest, dtype=str).fillna("")
    else:
        manifest = pd.DataFrame(columns=manifest_columns)
    for column in manifest_columns:
        if column not in manifest:
            manifest[column] = ""

    known_sources = set(manifest["source_file"].astype(str))
    additions = []
    for path in files:
        if path.name not in known_sources:
            additions.append(
                {
                    "source_file": path.name,
                    "headset_model": "Unknown",
                    "collection_round": "",
                    "notes": "",
                }
            )
    if additions:
        manifest = pd.concat([manifest, pd.DataFrame(additions)], ignore_index=True)
    manifest[manifest_columns].to_csv(config.source_manifest, index=False)

    review_columns = [
        "run_id",
        "headset_model",
        "session_type",
        "attempt_decision",
        "manual_exclusion_reason",
        "notes",
    ]
    if config.session_review.exists():
        review = pd.read_csv(config.session_review, dtype=str).fillna("")
    else:
        review = pd.DataFrame(columns=review_columns)
    for column in review_columns:
        if column not in review:
            review[column] = ""

    known_run_ids = set(review["run_id"].astype(str))
    new_rows = []
    for run_id in raw.get("runId", pd.Series(dtype=str)).dropna().astype(str).unique():
        if run_id and run_id not in known_run_ids:
            new_rows.append(
                {
                    "run_id": run_id,
                    "headset_model": "",
                    "session_type": "unknown",
                    "attempt_decision": "auto",
                    "manual_exclusion_reason": "",
                    "notes": "",
                }
            )
    if new_rows:
        review = pd.concat([review, pd.DataFrame(new_rows)], ignore_index=True)
    review[review_columns].to_csv(config.session_review, index=False)


def attach_review_metadata(raw: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    data = raw.copy()

    manifest = pd.read_csv(config.source_manifest, dtype=str).fillna("")
    manifest = manifest.drop_duplicates("source_file", keep="last")
    manifest_map = manifest.set_index("source_file")["headset_model"].to_dict()
    data["headset_from_source"] = data["source_file"].map(manifest_map).fillna("")

    review = pd.read_csv(config.session_review, dtype=str).fillna("")
    if review["run_id"].duplicated().any():
        duplicates = review.loc[review["run_id"].duplicated(False), "run_id"].tolist()
        raise ValueError(f"session_review.csv contains duplicate run_id rows: {duplicates}")

    review = review.rename(columns={"headset_model": "headset_override"})
    data = data.merge(review, how="left", left_on="runId", right_on="run_id")
    for column, default in {
        "headset_override": "",
        "session_type": "unknown",
        "attempt_decision": "auto",
        "manual_exclusion_reason": "",
        "notes": "",
    }.items():
        data[column] = data.get(column, default).fillna(default).astype(str).str.strip()

    data["headset_model"] = data["headset_override"].where(
        data["headset_override"].ne(""), data["headset_from_source"]
    )
    data["headset_model"] = data["headset_model"].replace("", "Unknown")
    return data


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _boolean(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def validate_records(
    data: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return issue rows and a copy of data with a structural-invalid flag."""
    issues: list[dict[str, Any]] = []

    def add(
        row: pd.Series,
        severity: str,
        field: str,
        issue: str,
        observed: Any = "",
        expected: Any = "",
    ) -> None:
        issues.append(
            {
                "severity": severity,
                "record_key": row.get("record_key", ""),
                "run_id": row.get("runId", ""),
                "participant_id": row.get("participantId", ""),
                "field": field,
                "issue": issue,
                "observed": observed,
                "expected": expected,
            }
        )

    required = [
        "runId",
        "participantId",
        "ddaAlgorithm",
        "dataCollectionConsent",
        "tutorialAnalytics.completed",
        "movementAnalytics.sampledDurationSeconds",
        "finalScore",
        "totalPatternsSpawned",
        "targetsHit",
        "targetsMissed",
        "totalShotsFired",
    ]

    seen_run_ids: set[str] = set()
    for _, row in data.iterrows():
        for field in required:
            value = row.get(field, np.nan)
            if pd.isna(value) or value == "":
                add(row, "error", field, "missing required value")

        run_id = str(row.get("runId", ""))
        if run_id in seen_run_ids and run_id:
            add(row, "error", "runId", "duplicate run identifier", run_id, "unique")
        seen_run_ids.add(run_id)

        strategy = row.get("ddaAlgorithm")
        if strategy not in STRATEGY_MAP:
            add(
                row,
                "error",
                "ddaAlgorithm",
                "unrecognized strategy identifier",
                strategy,
                ", ".join(STRATEGY_MAP),
            )

        if _boolean(row.get("dataCollectionConsent")) is not True:
            add(
                row,
                "error",
                "dataCollectionConsent",
                "recorded consent is not true",
                row.get("dataCollectionConsent"),
                True,
            )

        count_fields = [
            "finalScore",
            "totalPatternsSpawned",
            *PATTERN_COLUMNS.values(),
            "totalShotsFired",
            "redShotsFired",
            "blueShotsFired",
            "targetsHit",
            "targetsMissed",
            "correctColorHits",
            "wrongColorHits",
            "totalDepletedTimeSeconds",
            "wallContactTimeSeconds",
        ]
        for field in count_fields:
            value = _number(row.get(field))
            if np.isfinite(value) and value < 0:
                add(row, "error", field, "negative value", value, ">= 0")

        pattern_sum = sum(_number(row.get(field)) for field in PATTERN_COLUMNS.values())
        total_patterns = _number(row.get("totalPatternsSpawned"))
        if np.isfinite(pattern_sum) and np.isfinite(total_patterns) and pattern_sum != total_patterns:
            add(
                row,
                "error",
                "totalPatternsSpawned",
                "pattern-category counts do not sum to total",
                total_patterns,
                pattern_sum,
            )

        shots = _number(row.get("totalShotsFired"))
        shot_color_sum = _number(row.get("redShotsFired")) + _number(
            row.get("blueShotsFired")
        )
        if np.isfinite(shots) and np.isfinite(shot_color_sum) and shots != shot_color_sum:
            add(
                row,
                "error",
                "totalShotsFired",
                "red and blue shots do not sum to total shots",
                shots,
                shot_color_sum,
            )

        hits = _number(row.get("targetsHit"))
        hit_color_sum = _number(row.get("correctColorHits")) + _number(
            row.get("wrongColorHits")
        )
        if np.isfinite(hits) and np.isfinite(hit_color_sum) and hits != hit_color_sum:
            add(
                row,
                "error",
                "targetsHit",
                "correct- and wrong-color hits do not sum to targets hit",
                hits,
                hit_color_sum,
            )

        score = _number(row.get("finalScore"))
        expected_score = 10 * (
            _number(row.get("targetsHit")) + _number(row.get("correctColorHits"))
        )
        if np.isfinite(score) and np.isfinite(expected_score) and score != expected_score:
            add(
                row,
                "error",
                "finalScore",
                "score does not match 10 points per hit plus 10-point color bonus",
                score,
                expected_score,
            )

        stored_seen = _number(row.get("targetsSeen"))
        expected_seen = _number(row.get("targetsHit")) + _number(
            row.get("targetsMissed")
        )
        if np.isfinite(stored_seen) and stored_seen != expected_seen:
            add(
                row,
                "warning",
                "targetsSeen",
                "stored resolved-target count differs from raw-count calculation",
                stored_seen,
                expected_seen,
            )

        active = _number(row.get("movementAnalytics.sampledDurationSeconds"))
        wall_clock = _number(row.get("actualRunDurationSeconds"))
        if np.isfinite(active) and active <= 0:
            add(row, "error", "movementAnalytics.sampledDurationSeconds", "nonpositive duration", active, "> 0")
        if np.isfinite(active) and np.isfinite(wall_clock) and active > wall_clock + 0.25:
            add(
                row,
                "warning",
                "movementAnalytics.sampledDurationSeconds",
                "active sampled duration exceeds wall-clock duration",
                active,
                f"<= {wall_clock + 0.25:.3f}",
            )

        zero_time = _number(row.get("totalDepletedTimeSeconds"))
        wall_time = _number(row.get("wallContactTimeSeconds"))
        if np.isfinite(active) and np.isfinite(zero_time) and zero_time > active + 0.01:
            add(row, "error", "totalDepletedTimeSeconds", "exceeds active duration", zero_time, active)
        if np.isfinite(active) and np.isfinite(wall_time) and wall_time > active + 0.01:
            add(row, "error", "wallContactTimeSeconds", "exceeds active duration", wall_time, active)

        age = _number(row.get("age"))
        if config.expected_age_min is not None and np.isfinite(age) and age < config.expected_age_min:
            add(row, "warning", "age", "below expected recruitment range; verify session identity", age, f">= {config.expected_age_min}")
        if config.expected_age_max is not None and np.isfinite(age) and age > config.expected_age_max:
            add(row, "warning", "age", "above expected recruitment range; verify session identity", age, f"<= {config.expected_age_max}")

        experience = _number(row.get("experience"))
        if np.isfinite(experience) and int(experience) not in EXPERIENCE_MAP:
            add(row, "error", "experience", "unrecognized previous-VR-experience value", experience, "0, 1, or 2")

        ratings = [
            "postRunFeedback.difficultyRating",
            "postRunFeedback.enjoymentRating",
            "postRunFeedback.frustrationRating",
        ]
        valid_ratings = []
        for field in ratings:
            value = _number(row.get(field))
            valid = np.isfinite(value) and (value == 0 or 1 <= value <= 5)
            valid_ratings.append(np.isfinite(value) and 1 <= value <= 5)
            if not valid:
                add(row, "warning", field, "rating outside 0-5 range", value, "0-5")
        completed = _boolean(row.get("postRunFeedback.completed"))
        if completed is True and not all(valid_ratings):
            add(row, "warning", "postRunFeedback.completed", "feedback marked complete with one or more missing ratings")
        if completed is False and all(valid_ratings):
            add(row, "warning", "postRunFeedback.completed", "all ratings are valid but feedback is not marked complete")

        session_type = str(row.get("session_type", "unknown")).lower()
        if session_type not in ALLOWED_SESSION_TYPES:
            add(row, "warning", "session_type", "unrecognized review value", session_type, ", ".join(sorted(ALLOWED_SESSION_TYPES)))
        attempt = str(row.get("attempt_decision", "auto")).lower()
        if attempt not in ALLOWED_ATTEMPT_DECISIONS:
            add(row, "warning", "attempt_decision", "unrecognized review value", attempt, ", ".join(sorted(ALLOWED_ATTEMPT_DECISIONS)))
        reason = str(row.get("manual_exclusion_reason", ""))
        if reason not in ALLOWED_MANUAL_REASONS:
            add(row, "warning", "manual_exclusion_reason", "unrecognized review value", reason, ", ".join(sorted(ALLOWED_MANUAL_REASONS)))

    issue_frame = pd.DataFrame(
        issues,
        columns=[
            "severity",
            "record_key",
            "run_id",
            "participant_id",
            "field",
            "issue",
            "observed",
            "expected",
        ],
    )
    invalid_keys = set(
        issue_frame.loc[issue_frame["severity"].eq("error"), "record_key"]
    )
    checked = data.copy()
    checked["structurally_invalid"] = checked["record_key"].isin(invalid_keys)
    return issue_frame, checked


def _headset_key(value: Any) -> str:
    key = "".join(
        character for character in str(value).lower() if character.isalnum()
    )
    aliases = {
        "quest2": "metaquest2",
        "oculusquest2": "metaquest2",
        "quest3": "metaquest3",
    }
    return aliases.get(key, key)


def screen_records(
    checked: pd.DataFrame,
    config: AnalysisConfig,
    included_headsets: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Assign one mutually exclusive screening result to every exported record.

    The default retains only ``config.target_headset`` for the primary analysis.
    Passing ``included_headsets`` permits a separate sensitivity analysis without
    changing the primary-analysis configuration or review metadata.
    """
    data = checked.copy()
    data["exclusion_reason"] = ""

    if included_headsets is None:
        included_headsets = (config.target_headset,)
    included_headset_keys = {
        _headset_key(headset) for headset in included_headsets
    }
    if not included_headset_keys or included_headset_keys & {"", "unknown"}:
        raise ValueError("included_headsets must contain known headset models")

    for index, row in data.iterrows():
        manual_reason = str(row.get("manual_exclusion_reason", "")).strip()
        session_type = str(row.get("session_type", "unknown")).strip().lower()

        if manual_reason in ALLOWED_MANUAL_REASONS and manual_reason:
            data.at[index, "exclusion_reason"] = manual_reason
            continue
        if str(row.get("attempt_decision", "auto")).strip().lower() == "exclude":
            data.at[index, "exclusion_reason"] = "duplicate_or_invalid_record"
            continue
        if session_type in NONPARTICIPANT_SESSION_TYPES:
            data.at[index, "exclusion_reason"] = (
                "researcher_staff_demo_or_informal"
            )
            continue
        if bool(row.get("structurally_invalid", False)):
            data.at[index, "exclusion_reason"] = "duplicate_or_invalid_record"
            continue

        active = _number(row.get("movementAnalytics.sampledDurationSeconds"))
        tutorial_complete = _boolean(row.get("tutorialAnalytics.completed"))
        duration_ok = (
            np.isfinite(active)
            and active >= config.active_duration_min_s
        )
        if tutorial_complete is not True or not duration_ok:
            data.at[index, "exclusion_reason"] = "interrupted_or_incomplete"
            continue

        headset = _headset_key(row.get("headset_model", "Unknown"))
        if headset == "metaquest2" and headset not in included_headset_keys:
            data.at[index, "exclusion_reason"] = "eligible_meta_quest_2"
        elif headset not in {"", "unknown"} and headset not in included_headset_keys:
            data.at[index, "exclusion_reason"] = "other_ineligible"

    candidates = data[data["exclusion_reason"].eq("")].copy()
    candidates["_started"] = pd.to_datetime(
        candidates.get("startedUtc"), errors="coerce", utc=True
    )
    candidates["_attempt"] = pd.to_numeric(
        candidates.get("attemptIndex"), errors="coerce"
    ).fillna(np.inf)

    for _, group in candidates.groupby("participantId", dropna=False):
        if len(group) <= 1:
            continue
        keep_rows = group[
            group["attempt_decision"].astype(str).str.lower().eq("keep")
        ]
        if len(keep_rows) > 1:
            ids = keep_rows.get("runId", pd.Series(dtype=str)).tolist()
            raise ValueError(
                "More than one attempt is marked keep for one participant: "
                + ", ".join(map(str, ids))
            )
        if len(keep_rows) == 1:
            keep_index = keep_rows.index[0]
        else:
            ordered = group.sort_values(
                ["_attempt", "_started", "source_file", "source_row"],
                na_position="last",
            )
            keep_index = ordered.index[0]
        exclude_indices = [item for item in group.index if item != keep_index]
        data.loc[exclude_indices, "exclusion_reason"] = (
            "duplicate_or_invalid_record"
        )

    unresolved_mask = data["exclusion_reason"].eq("") & (
        data["session_type"].astype(str).str.lower().eq("unknown")
        | data["session_type"].astype(str).str.strip().eq("")
        | ~data["session_type"].astype(str).str.lower().isin(
            ALLOWED_SESSION_TYPES
        )
        | ~data["attempt_decision"].astype(str).str.lower().isin(
            ALLOWED_ATTEMPT_DECISIONS
        )
        | ~data["manual_exclusion_reason"].astype(str).isin(
            ALLOWED_MANUAL_REASONS
        )
        | data["headset_model"].map(_headset_key).isin({"", "unknown"})
        | ~data["headset_model"].map(_headset_key).isin(
            {"", "unknown", *included_headset_keys}
        )
    )
    unresolved_ids = (
        data.loc[unresolved_mask, "runId"].dropna().astype(str).tolist()
    )
    if config.strict_manual_review and unresolved_ids:
        raise ReviewRequiredError(
            "Final-mode analysis is blocked. Complete headset_model and "
            "session_type in review/session_review.csv for: "
            + ", ".join(unresolved_ids)
        )

    data["screening_status"] = np.where(
        data["exclusion_reason"].ne(""),
        "excluded",
        np.where(unresolved_mask, "provisional_include", "included"),
    )
    data.loc[data["exclusion_reason"].eq(""), "exclusion_reason"] = "included"

    counts = data["exclusion_reason"].value_counts().to_dict()
    flow = pd.DataFrame(
        {
            "screening_category": FLOW_ORDER,
            "description": [FLOW_LABELS[item] for item in FLOW_ORDER],
            "n": [int(counts.get(item, 0)) for item in FLOW_ORDER],
        }
    )
    return data, flow, unresolved_ids


def derive_outcomes(screened: pd.DataFrame) -> pd.DataFrame:
    data = screened.loc[
        screened["screening_status"].isin({"included", "provisional_include"})
    ].copy()

    def numeric(column: str) -> pd.Series:
        return pd.to_numeric(data.get(column), errors="coerce")

    data["strategy"] = data["ddaAlgorithm"].map(STRATEGY_MAP)
    data["experience_category"] = numeric("experience").map(EXPERIENCE_MAP)
    data["active_duration_s"] = numeric(
        "movementAnalytics.sampledDurationSeconds"
    )
    data["wall_clock_duration_s"] = numeric("actualRunDurationSeconds")
    data["final_score"] = numeric("finalScore")
    data["total_patterns_spawned"] = numeric("totalPatternsSpawned")
    data["resolved_targets"] = numeric("targetsHit") + numeric("targetsMissed")

    with np.errstate(divide="ignore", invalid="ignore"):
        data["normalized_target_score_pct"] = (
            data["final_score"] / (20.0 * data["resolved_targets"]) * 100.0
        )
        data["target_hit_rate_pct"] = (
            numeric("targetsHit") / data["resolved_targets"] * 100.0
        )
        data["shot_accuracy_pct"] = (
            numeric("targetsHit") / numeric("totalShotsFired") * 100.0
        )
        data["color_match_rate_pct"] = (
            numeric("correctColorHits") / numeric("targetsHit") * 100.0
        )
        data["wall_contact_pct"] = (
            numeric("wallContactTimeSeconds") / data["active_duration_s"] * 100.0
        )
        data["zero_stability_pct"] = (
            numeric("totalDepletedTimeSeconds") / data["active_duration_s"] * 100.0
        )
        data["hard_expert_pattern_pct"] = (
            (numeric("hardPatternsSpawned") + numeric("expertPatternsSpawned"))
            / data["total_patterns_spawned"]
            * 100.0
        )
        data["horizontal_movement_m_per_min"] = (
            numeric("movementAnalytics.headHorizontalDistance")
            / data["active_duration_s"]
            * 60.0
        )

    for label, source in PATTERN_COLUMNS.items():
        destination = f"pattern_{label.lower().replace(' ', '_')}_pct"
        with np.errstate(divide="ignore", invalid="ignore"):
            data[destination] = (
                numeric(source) / data["total_patterns_spawned"] * 100.0
            )

    data["maximum_scroll_speed_m_s"] = numeric(
        "actualMaxScrollSpeedReached"
    )
    data["minimum_pattern_gap_m"] = numeric("actualMinPatternGapReached")

    for source, destination in [
        ("postRunFeedback.difficultyRating", "difficulty_rating"),
        ("postRunFeedback.enjoymentRating", "enjoyment_rating"),
        ("postRunFeedback.frustrationRating", "frustration_rating"),
    ]:
        values = numeric(source)
        data[destination] = values.where(values.between(1, 5))
    data["difficulty_deviation"] = (data["difficulty_rating"] - 3).abs()

    feedback_flag = data.get(
        "postRunFeedback.completed", pd.Series(False, index=data.index)
    ).map(_boolean)
    data["feedback_complete"] = (
        feedback_flag.eq(True)
        & data[
            ["difficulty_rating", "enjoyment_rating", "frustration_rating"]
        ].notna().all(axis=1)
    )
    motion = data.get(
        "postRunFeedback.motionSickness", pd.Series(False, index=data.index)
    ).map(_boolean)
    data["motion_sickness_report"] = motion.where(data["feedback_complete"])

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data
