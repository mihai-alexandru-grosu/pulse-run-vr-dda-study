from __future__ import annotations

import argparse
from pathlib import Path

from pulse_run_analysis import (
    refresh_review_templates,
    run_analysis,
    run_complete_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Pulse Run VR Section 4 analysis."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("analysis_config.toml"),
        help="Path to the TOML configuration file.",
    )
    parser.add_argument(
        "--init-review",
        action="store_true",
        help="Create or update the source and session review CSV files, then stop.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Run only the Quest 3 primary analysis and skip the sensitivity check.",
    )
    args = parser.parse_args()

    if args.init_review:
        manifest, review = refresh_review_templates(args.config)
        print(f"Updated: {manifest}")
        print(f"Updated: {review}")
        return

    if args.primary_only:
        primary = run_analysis(args.config)
        sensitivity = None
    else:
        complete = run_complete_analysis(args.config)
        primary = complete.primary
        sensitivity = complete.sensitivity

    print(f"Primary analysis status: {primary.status['analysis_status']}")
    print(f"Raw records: {primary.status['raw_records']}")
    print(f"Quest 3 primary sessions: {primary.status['analyzed_sessions']}")
    print(f"Primary outputs: {primary.output_dir}")
    if sensitivity is not None:
        print(
            "All-headset sensitivity sessions: "
            f"{sensitivity.status['analyzed_sessions']}"
        )
        print(f"Sensitivity outputs: {sensitivity.output_dir}")
    if primary.status["unresolved_review_records"]:
        print(
            "Review metadata remains unresolved for "
            f"{primary.status['unresolved_review_records']} otherwise eligible records."
        )


if __name__ == "__main__":
    main()
