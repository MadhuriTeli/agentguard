#!/usr/bin/env python
"""Compare two sets of run results and print a regression report.

Usage:
    python scripts/compare_runs.py --baseline baseline.json --candidate candidate.json

Each input file is expected to be a JSON list of RunResult summaries
(as produced by RunResult.summary()).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentguard.core.result import EvalResult, RunResult, Verdict
from agentguard.regression.comparator import RegressionComparator


def load_run_results(path: Path) -> list[RunResult]:
    data = json.loads(path.read_text())
    runs = []
    for entry in data:
        runs.append(
            RunResult(
                trajectory_id=entry.get("trajectory_id", ""),
                agent_name=entry.get("agent_name", ""),
                scenario_name=entry["scenario_name"],
                eval_results=[
                    EvalResult(name=r["name"], verdict=Verdict(r["verdict"]))
                    for r in entry.get("results", [])
                ],
            )
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()

    baseline = load_run_results(args.baseline)
    candidate = load_run_results(args.candidate)

    report = RegressionComparator().compare(baseline, candidate)

    print(f"Regressions: {len(report.regressions)}")
    for entry in report.regressions:
        print(f"  - {entry.scenario_name} / {entry.eval_name}: "
              f"{entry.baseline_verdict} -> {entry.candidate_verdict}")

    print(f"Improvements: {len(report.improvements)}")
    for entry in report.improvements:
        print(f"  - {entry.scenario_name} / {entry.eval_name}: "
              f"{entry.baseline_verdict} -> {entry.candidate_verdict}")

    if report.has_regressions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
