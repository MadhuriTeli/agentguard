"""Suite-level, metric-based regression comparison.

RegressionComparator (comparator.py) answers "did any individual eval
verdict flip from pass to fail" — a binary, per-scenario signal used by
Harness itself to gate a single run against a baseline.

This module answers a different, complementary question: "did the
CANDIDATE'S AGGREGATE BEHAVIOR get worse across a whole suite" — pass
rate, tool-call efficiency, latency, and failure-category counts. A
candidate can pass every individual verdict check and still be a worse
agent (three times as many tool calls, much higher latency) — this is how
you catch that.

Deliberately NOT included: token usage / cost. Nothing in this codebase
currently measures tokens consumed by an agent run (the agent's own
reasoning is a black box to Harness; only tool calls and their timing are
observable), so a token metric here would be fabricated. Add it once
there's a real source for it — an LLM-backed agent adapter that reports
its own usage, for instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentguard.core.harness import HarnessResult
from agentguard.core.result import FailureCategory, FailureType, category_of


def _all_failure_types(run: HarnessResult) -> list[FailureType]:
    types: list[FailureType] = []
    if run.contract_result is not None:
        types.extend(run.contract_result.failure_types)
    for eval_result in run.run_result.eval_results:
        types.extend(eval_result.failure_types)
    return types


def _failure_categories(run: HarnessResult) -> set[FailureCategory]:
    return {category_of(ft) for ft in _all_failure_types(run)}


@dataclass
class SuiteMetrics:
    """Aggregate metrics across every HarnessResult in one suite run."""

    num_scenarios: int
    num_passed: int
    total_tool_calls: int
    total_duration_seconds: float
    contract_failures: int
    safety_failures: int

    @property
    def pass_rate(self) -> float:
        return self.num_passed / self.num_scenarios if self.num_scenarios else 0.0

    @property
    def avg_tool_calls(self) -> float:
        return self.total_tool_calls / self.num_scenarios if self.num_scenarios else 0.0

    @property
    def avg_duration_seconds(self) -> float:
        return self.total_duration_seconds / self.num_scenarios if self.num_scenarios else 0.0

    @classmethod
    def from_runs(cls, runs: list[HarnessResult]) -> SuiteMetrics:
        contract_failures = 0
        safety_failures = 0
        for run in runs:
            categories = _failure_categories(run)
            if FailureCategory.CONTRACT_FAILURE in categories:
                contract_failures += 1
            if FailureCategory.SAFETY_FAILURE in categories:
                safety_failures += 1

        return cls(
            num_scenarios=len(runs),
            num_passed=sum(1 for r in runs if r.passed),
            total_tool_calls=sum(r.metrics.num_tool_calls for r in runs),
            total_duration_seconds=sum(r.metrics.duration_seconds or 0.0 for r in runs),
            contract_failures=contract_failures,
            safety_failures=safety_failures,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "num_scenarios": self.num_scenarios,
            "num_passed": self.num_passed,
            "pass_rate": self.pass_rate,
            "avg_tool_calls": self.avg_tool_calls,
            "avg_duration_seconds": self.avg_duration_seconds,
            "contract_failures": self.contract_failures,
            "safety_failures": self.safety_failures,
        }


@dataclass
class MetricRegression:
    """One metric compared between baseline and candidate, with a
    threshold determining whether the change counts as a regression.

    direction is "higher_is_worse" (tool calls, latency, failure counts)
    or "lower_is_worse" (pass rate) — the two shapes every metric here
    actually needs; extend with a real comparator function if a future
    metric doesn't fit either.
    """

    metric_name: str
    baseline_value: float
    candidate_value: float
    threshold: float
    direction: str

    @property
    def delta(self) -> float:
        return self.candidate_value - self.baseline_value

    @property
    def is_regression(self) -> bool:
        if self.direction == "higher_is_worse":
            return self.delta > self.threshold
        return self.delta < -self.threshold


@dataclass
class SuiteRegressionReport:
    baseline: SuiteMetrics
    candidate: SuiteMetrics
    metric_regressions: list[MetricRegression] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return any(m.is_regression for m in self.metric_regressions)

    @property
    def regressions(self) -> list[MetricRegression]:
        return [m for m in self.metric_regressions if m.is_regression]

    def summary_table(self) -> list[dict[str, float | str]]:
        return [
            {
                "metric": m.metric_name,
                "baseline": m.baseline_value,
                "candidate": m.candidate_value,
                "regression": m.is_regression,
            }
            for m in self.metric_regressions
        ]


class MetricsComparator:
    """Compares aggregate suite-level metrics between a baseline and
    candidate batch of HarnessResults.

    Thresholds default to zero tolerance for pass rate, tool calls, and
    failure counts — any worsening counts as a regression there. Latency
    is the one exception: wall-clock duration has inherent measurement
    noise (two runs of identical code will essentially never take exactly
    the same number of microseconds), so a zero-tolerance latency
    threshold would flag noise as a regression on every single comparison.
    It defaults to 50ms instead — enough to absorb ordinary jitter while
    still catching a real slowdown (the kind of +45% latency swing this
    metric exists to catch is measured in hundreds of milliseconds, not
    microseconds).
    """

    def __init__(
        self,
        pass_rate_drop_threshold: float = 0.0,
        tool_call_increase_threshold: float = 0.0,
        latency_increase_threshold_seconds: float = 0.05,
    ) -> None:
        self.pass_rate_drop_threshold = pass_rate_drop_threshold
        self.tool_call_increase_threshold = tool_call_increase_threshold
        self.latency_increase_threshold_seconds = latency_increase_threshold_seconds

    def compare(
        self, baseline_runs: list[HarnessResult], candidate_runs: list[HarnessResult]
    ) -> SuiteRegressionReport:
        baseline = SuiteMetrics.from_runs(baseline_runs)
        candidate = SuiteMetrics.from_runs(candidate_runs)

        metric_regressions = [
            MetricRegression(
                "pass_rate",
                baseline.pass_rate,
                candidate.pass_rate,
                self.pass_rate_drop_threshold,
                "lower_is_worse",
            ),
            MetricRegression(
                "avg_tool_calls",
                baseline.avg_tool_calls,
                candidate.avg_tool_calls,
                self.tool_call_increase_threshold,
                "higher_is_worse",
            ),
            MetricRegression(
                "avg_duration_seconds",
                baseline.avg_duration_seconds,
                candidate.avg_duration_seconds,
                self.latency_increase_threshold_seconds,
                "higher_is_worse",
            ),
            MetricRegression(
                "contract_failures",
                baseline.contract_failures,
                candidate.contract_failures,
                0,
                "higher_is_worse",
            ),
            MetricRegression(
                "safety_failures",
                baseline.safety_failures,
                candidate.safety_failures,
                0,
                "higher_is_worse",
            ),
        ]

        return SuiteRegressionReport(
            baseline=baseline, candidate=candidate, metric_regressions=metric_regressions
        )
