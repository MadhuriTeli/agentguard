"""Compares evaluation results across two runs (e.g. baseline vs candidate)
to detect behavioral regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentguard.core.result import RunResult


@dataclass
class RegressionEntry:
    scenario_name: str
    eval_name: str
    baseline_verdict: str
    candidate_verdict: str

    @property
    def is_regression(self) -> bool:
        return self.baseline_verdict == "pass" and self.candidate_verdict != "pass"

    @property
    def is_improvement(self) -> bool:
        return self.baseline_verdict != "pass" and self.candidate_verdict == "pass"


@dataclass
class RegressionReport:
    entries: list[RegressionEntry] = field(default_factory=list)

    @property
    def regressions(self) -> list[RegressionEntry]:
        return [e for e in self.entries if e.is_regression]

    @property
    def improvements(self) -> list[RegressionEntry]:
        return [e for e in self.entries if e.is_improvement]

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0


class RegressionComparator:
    """Diffs two sets of RunResults (baseline vs candidate) by scenario."""

    def compare(
        self, baseline: list[RunResult], candidate: list[RunResult]
    ) -> RegressionReport:
        baseline_by_scenario = {r.scenario_name: r for r in baseline}
        candidate_by_scenario = {r.scenario_name: r for r in candidate}

        entries: list[RegressionEntry] = []
        common_scenarios = set(baseline_by_scenario) & set(candidate_by_scenario)

        for scenario_name in sorted(common_scenarios):
            base_run = baseline_by_scenario[scenario_name]
            cand_run = candidate_by_scenario[scenario_name]

            base_by_eval = {r.name: r for r in base_run.eval_results}
            cand_by_eval = {r.name: r for r in cand_run.eval_results}

            for eval_name in set(base_by_eval) & set(cand_by_eval):
                entries.append(
                    RegressionEntry(
                        scenario_name=scenario_name,
                        eval_name=eval_name,
                        baseline_verdict=base_by_eval[eval_name].verdict.value,
                        candidate_verdict=cand_by_eval[eval_name].verdict.value,
                    )
                )

        return RegressionReport(entries=entries)
