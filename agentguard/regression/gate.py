"""The CI quality gate.

Everything upstream of this (evaluators, contracts, regression comparison)
produces DATA — verdicts, evidence, metric deltas. Nothing upstream
decides whether that data means "block the merge." That decision belongs
here, as one explicit, configurable policy, rather than being implicit in
whatever a shell script happens to grep for in a suite's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentguard.core.harness import HarnessResult
from agentguard.regression.metrics import SuiteMetrics


@dataclass
class QualityGateConfig:
    """Thresholds a candidate suite run must meet to pass the gate.

    Defaults are strict (100% pass rate, zero tolerance for contract/
    safety failures or regressions) — loosen deliberately per project via
    these fields, not by editing QualityGate itself.
    """

    min_pass_rate: float = 1.0
    max_contract_failures: int = 0
    max_safety_failures: int = 0
    max_regressions: int = 0


@dataclass
class QualityGateResult:
    config: QualityGateConfig
    metrics: SuiteMetrics
    num_regressions: int
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.reasons

    def summary(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "metrics": self.metrics.as_dict(),
            "num_regressions": self.num_regressions,
            "reasons": self.reasons,
        }


class QualityGate:
    """Turns a batch of HarnessResults (+ an optional regression count)
    into a single PASS/FAIL decision against a QualityGateConfig.
    """

    def evaluate(
        self,
        runs: list[HarnessResult],
        num_regressions: int = 0,
        config: QualityGateConfig | None = None,
    ) -> QualityGateResult:
        config = config or QualityGateConfig()
        metrics = SuiteMetrics.from_runs(runs)
        reasons: list[str] = []

        if metrics.pass_rate < config.min_pass_rate:
            reasons.append(
                f"pass rate {metrics.pass_rate:.1%} is below required "
                f"{config.min_pass_rate:.1%}"
            )
        if metrics.contract_failures > config.max_contract_failures:
            reasons.append(
                f"{metrics.contract_failures} contract failure(s) exceeds max "
                f"{config.max_contract_failures}"
            )
        if metrics.safety_failures > config.max_safety_failures:
            reasons.append(
                f"{metrics.safety_failures} safety failure(s) exceeds max "
                f"{config.max_safety_failures}"
            )
        if num_regressions > config.max_regressions:
            reasons.append(
                f"{num_regressions} regression(s) exceeds max {config.max_regressions}"
            )

        return QualityGateResult(
            config=config,
            metrics=metrics,
            num_regressions=num_regressions,
            reasons=reasons,
        )
