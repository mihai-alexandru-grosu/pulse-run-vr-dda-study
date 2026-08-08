from __future__ import annotations

from dataclasses import dataclass


STRATEGY_MAP = {
    "FixedTime": "Fixed Time-Based",
    "PerformanceDda": "Performance-Based DDA",
    "SkillSplitDda": "Skill-Specific DDA",
}

STRATEGY_ORDER = [
    "Fixed Time-Based",
    "Performance-Based DDA",
    "Skill-Specific DDA",
]

STRATEGY_SHORT = {
    "Fixed Time-Based": "Fixed Time-Based",
    "Performance-Based DDA": "Performance-Based DDA",
    "Skill-Specific DDA": "Skill-Specific DDA",
}

EXPERIENCE_MAP = {
    0: "None",
    1: "Some",
    2: "High",
}

PATTERN_COLUMNS = {
    "Very Easy": "veryEasyPatternsSpawned",
    "Easy": "easyPatternsSpawned",
    "Medium": "mediumPatternsSpawned",
    "Hard": "hardPatternsSpawned",
    "Expert": "expertPatternsSpawned",
}

PATTERN_PERCENT_COLUMNS = {
    label: f"pattern_{label.lower().replace(' ', '_')}_pct"
    for label in PATTERN_COLUMNS
}

PRIMARY_OUTCOMES = [
    "normalized_target_score_pct",
    "wall_contact_pct",
    "hard_expert_pattern_pct",
    "difficulty_deviation",
    "enjoyment_rating",
]

SECONDARY_OUTCOMES = [
    "final_score",
    "target_hit_rate_pct",
    "shot_accuracy_pct",
    "color_match_rate_pct",
    "zero_stability_pct",
    "total_patterns_spawned",
    "resolved_targets",
    "frustration_rating",
]

DESCRIPTIVE_OUTCOMES = [
    "maximum_scroll_speed_m_s",
    "minimum_pattern_gap_m",
    *PATTERN_PERCENT_COLUMNS.values(),
    "difficulty_rating",
    "motion_sickness_report",
]


@dataclass(frozen=True)
class OutcomeSpec:
    key: str
    label: str
    test: str
    role: str
    section: str
    digits: int = 1


OUTCOME_SPECS = [
    OutcomeSpec(
        "hard_expert_pattern_pct",
        "Hard/Expert patterns, %",
        "welch",
        "primary",
        "pacing",
    ),
    OutcomeSpec(
        "total_patterns_spawned",
        "Total patterns spawned",
        "welch",
        "secondary",
        "pacing",
    ),
    OutcomeSpec(
        "resolved_targets",
        "Resolved targets",
        "welch",
        "secondary",
        "pacing",
    ),
    OutcomeSpec(
        "normalized_target_score_pct",
        "Normalized target score, %",
        "welch",
        "primary",
        "performance",
    ),
    OutcomeSpec(
        "wall_contact_pct",
        "Wall-contact percentage",
        "welch",
        "primary",
        "performance",
    ),
    OutcomeSpec(
        "final_score",
        "Final score",
        "welch",
        "secondary",
        "performance",
    ),
    OutcomeSpec(
        "target_hit_rate_pct",
        "Target hit rate, %",
        "welch",
        "secondary",
        "performance",
    ),
    OutcomeSpec(
        "shot_accuracy_pct",
        "Shot accuracy, %",
        "welch",
        "secondary",
        "performance",
    ),
    OutcomeSpec(
        "color_match_rate_pct",
        "Color-match rate, %",
        "welch",
        "secondary",
        "performance",
    ),
    OutcomeSpec(
        "zero_stability_pct",
        "Zero-stability percentage",
        "kruskal",
        "secondary",
        "performance",
    ),
    OutcomeSpec(
        "difficulty_deviation",
        "Difficulty deviation",
        "kruskal",
        "primary",
        "ratings",
    ),
    OutcomeSpec(
        "enjoyment_rating",
        "Enjoyment",
        "kruskal",
        "primary",
        "ratings",
    ),
    OutcomeSpec(
        "frustration_rating",
        "Frustration",
        "kruskal",
        "secondary",
        "ratings",
    ),
]

OUTCOME_BY_KEY = {spec.key: spec for spec in OUTCOME_SPECS}

FLOW_ORDER = [
    "researcher_staff_demo_or_informal",
    "interrupted_or_incomplete",
    "duplicate_or_invalid_record",
    "other_ineligible",
    "eligible_meta_quest_2",
    "included",
]

FLOW_LABELS = {
    "researcher_staff_demo_or_informal": (
        "Researcher, staff, demonstration, or informal session"
    ),
    "interrupted_or_incomplete": "Interrupted or incomplete session",
    "duplicate_or_invalid_record": "Duplicate or invalid record",
    "other_ineligible": "Other ineligible record",
    "eligible_meta_quest_2": "Otherwise eligible Meta Quest 2 session",
    "included": "Included in primary analysis",
}

NONPARTICIPANT_SESSION_TYPES = {
    "researcher_test",
    "staff",
    "demonstration",
    "informal_play",
}

ALLOWED_SESSION_TYPES = {"participant", *NONPARTICIPANT_SESSION_TYPES, "unknown"}
ALLOWED_ATTEMPT_DECISIONS = {"auto", "keep", "exclude"}
ALLOWED_MANUAL_REASONS = {
    "",
    "researcher_staff_demo_or_informal",
    "interrupted_or_incomplete",
    "duplicate_or_invalid_record",
    "other_ineligible",
}
