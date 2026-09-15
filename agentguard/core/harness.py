"""The AgentGuard Harness.

This is the single entry point that ties every other piece together:

    Agent
      |
    Harness
      |-- intercept tools       (wraps each tool fn the agent can call)
      |-- enforce contracts     (blocks forbidden/over-quota calls live)
      |-- record trajectory     (every call, result, fault, violation)
      |-- inject failures       (via a ChaosEngine, at the interception point)
      |-- evaluate result       (contract validator + judges)
      |-- calculate metrics     (latency, call counts, fault/violation counts)
      \\-- detect regressions    (vs an optional baseline RunResult)
      |
    PASS / FAIL

Where the earlier pieces (Runner, ContractValidator, ChaosEngine,
EvaluationPipeline, RegressionComparator) are independent building blocks
you can use standalone, the Harness is the opinionated orchestration layer
most users will actually reach for.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agentguard.chaos.engine import ChaosEngine
from agentguard.contracts.schema import Contract
from agentguard.contracts.validator import ContractValidator
from agentguard.core.evaluator import EvaluationPipeline, Evaluator
from agentguard.core.result import EvalResult, RunResult, Verdict
from agentguard.core.runner import Scenario
from agentguard.core.trajectory import StepType, Trajectory
from agentguard.regression.comparator import RegressionComparator, RegressionEntry


class ContractViolation(Exception):
    """Raised when a live tool call breaks a contract's real-time rules."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class HarnessMetrics:
    """Quantitative summary of a single harness run."""

    duration_seconds: Optional[float]
    num_steps: int
    num_tool_calls: int
    tool_call_counts: dict[str, int]
    num_faults_injected: int
    num_live_violations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "num_steps": self.num_steps,
            "num_tool_calls": self.num_tool_calls,
            "tool_call_counts": dict(self.tool_call_counts),
            "num_faults_injected": self.num_faults_injected,
            "num_live_violations": self.num_live_violations,
        }


@dataclass
class HarnessResult:
    """Everything the harness produced for one run, plus the final verdict."""

    trajectory: Trajectory
    run_result: RunResult
    contract_result: Optional[EvalResult]
    metrics: HarnessMetrics
    regression_entries: list[RegressionEntry] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        contract_ok = self.contract_result is None or self.contract_result.passed
        evals_ok = self.run_result.passed
        no_regressions = not any(e.is_regression for e in self.regression_entries)
        return contract_ok and evals_ok and no_regressions

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def summary(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "trajectory_id": self.trajectory.id,
            "scenario_name": self.trajectory.scenario_name,
            "contract": (
                {"verdict": self.contract_result.verdict.value, "reason": self.contract_result.reason}
                if self.contract_result
                else None
            ),
            "eval_results": self.run_result.summary()["results"],
            "metrics": self.metrics.as_dict(),
            "regressions": [
                {
                    "eval_name": e.eval_name,
                    "baseline": e.baseline_verdict,
                    "candidate": e.candidate_verdict,
                }
                for e in self.regression_entries
                if e.is_regression
            ],
        }


class Harness:
    """Wraps an agent + its tools and runs one scenario end-to-end.

    Parameters
    ----------
    agent:
        Object exposing `step(observation) -> {"tool": str, "args": dict}`.
    tools:
        Mapping of tool name -> callable(**args) -> observation. The harness
        instruments each of these; the agent should never call the raw tool
        directly.
    contract:
        Optional Contract enforced both live (forbidden tools, max_calls)
        and post-hoc (required tools, max_turns) via ContractValidator.
    chaos_engine:
        Optional ChaosEngine applied to each tool result before it's
        returned to the agent.
    evaluators:
        Judges/evaluators run against the finished trajectory.
    baseline:
        Optional prior RunResult for the same scenario, used to detect
        regressions in this run's evaluator verdicts.
    """

    def __init__(
        self,
        agent: Any,
        tools: dict[str, Callable[..., Any]],
        contract: Optional[Contract] = None,
        chaos_engine: Optional[ChaosEngine] = None,
        evaluators: Optional[list[Evaluator]] = None,
        baseline: Optional[RunResult] = None,
    ):
        self.agent = agent
        self.contract = contract
        self.chaos_engine = chaos_engine
        self.evaluation_pipeline = EvaluationPipeline(evaluators or [])
        self.baseline = baseline

        self._tool_call_counts: Counter[str] = Counter()
        self._live_violations: list[str] = []
        self._trajectory: Optional[Trajectory] = None

        self.tools = {name: self._intercept(name, fn) for name, fn in tools.items()}

    # -- tool interception ------------------------------------------------

    def _intercept(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(**kwargs: Any) -> Any:
            assert self._trajectory is not None, "intercepted tool called outside a run"
            trajectory = self._trajectory

            self._tool_call_counts[name] += 1
            self._enforce_live(name, trajectory)

            try:
                result = fn(**kwargs)
            except Exception as exc:  # noqa: BLE001
                trajectory.add_step(StepType.ERROR, f"{name} raised: {exc}")
                raise

            if self.chaos_engine is not None:
                result = self.chaos_engine.maybe_inject(result, trajectory)

            trajectory.add_step(StepType.TOOL_RESULT, result, tool=name)

            # A reply tool represents the agent's user-facing response.
            # Record it as an assistant MESSAGE so trajectory-level
            # evaluators can inspect the final response.
            if name == "reply":
                message = (
                    result.get("message", "")
                    if isinstance(result, dict)
                    else str(result)
                )

                trajectory.add_step(
                    StepType.MESSAGE,
                    message,
                    role="assistant",
                )

            return result

        return wrapped

    def _enforce_live(self, tool_name: str, trajectory: Trajectory) -> None:
        """Block/record contract violations that are knowable at call time."""
        if self.contract is None:
            return

        if self.contract.forbidden_tools and tool_name in self.contract.forbidden_tools:
            msg = f"Forbidden tool called: {tool_name}"
            self._live_violations.append(msg)
            trajectory.add_step(StepType.ERROR, msg, kind="contract_violation")
            raise ContractViolation(msg)

        for constraint in self.contract.tool_constraints:
            if constraint.tool_name != tool_name or constraint.max_calls is None:
                continue
            if self._tool_call_counts[tool_name] > constraint.max_calls:
                msg = (
                    f"Tool '{tool_name}' exceeded max_calls "
                    f"({self._tool_call_counts[tool_name]} > {constraint.max_calls})"
                )
                self._live_violations.append(msg)
                trajectory.add_step(StepType.ERROR, msg, kind="contract_violation")
                raise ContractViolation(msg)

    # -- orchestration ------------------------------------------------------

    def run(self, scenario: Scenario) -> HarnessResult:
        self._tool_call_counts = Counter()
        self._live_violations = []
        trajectory = Trajectory(
            agent_name=getattr(self.agent, "name", self.agent.__class__.__name__),
            scenario_name=scenario.name,
        )
        self._trajectory = trajectory

        observation: Any = scenario.initial_input
        trajectory.add_step(StepType.MESSAGE, observation, role="user")

        success = False
        try:
            for _ in range(scenario.max_steps):
                action = self.agent.step(observation)
                trajectory.add_step(StepType.TOOL_CALL, action)

                tool_name = action.get("tool") if isinstance(action, dict) else None
                args = action.get("args", {}) if isinstance(action, dict) else {}

                if tool_name not in self.tools:
                    trajectory.add_step(
                        StepType.ERROR, f"Unknown tool requested: {tool_name}"
                    )
                    break

                observation = self.tools[tool_name](**args)

                if scenario.is_done and scenario.is_done(trajectory):
                    success = True
                    break
            else:
                success = False
        except ContractViolation:
            success = False
        except Exception as exc:  # noqa: BLE001
            trajectory.add_step(StepType.ERROR, str(exc))
            success = False

        trajectory.finish(success=success and not self._live_violations)
        self._trajectory = None

        contract_result = self._evaluate_contract(trajectory)
        run_result = self.evaluation_pipeline.run(trajectory, self.contract)
        metrics = self._calculate_metrics(trajectory)
        regression_entries = self._detect_regressions(run_result)

        return HarnessResult(
            trajectory=trajectory,
            run_result=run_result,
            contract_result=contract_result,
            metrics=metrics,
            regression_entries=regression_entries,
        )

    # -- post-run steps -----------------------------------------------------

    def _evaluate_contract(self, trajectory: Trajectory) -> Optional[EvalResult]:
        if self.contract is None:
            return None

        result = ContractValidator().validate(trajectory, self.contract)

        if self._live_violations:
            result = EvalResult(
                name=result.name,
                verdict=Verdict.FAIL,
                reason="; ".join(self._live_violations + [result.reason]),
                details={
                    "live_violations": self._live_violations,
                    **result.details,
                },
            )
        return result

    def _calculate_metrics(self, trajectory: Trajectory) -> HarnessMetrics:
        return HarnessMetrics(
            duration_seconds=trajectory.duration_seconds,
            num_steps=len(trajectory.steps),
            num_tool_calls=sum(self._tool_call_counts.values()),
            tool_call_counts=dict(self._tool_call_counts),
            num_faults_injected=sum(
                1 for s in trajectory.steps if s.type == StepType.FAULT_INJECTED
            ),
            num_live_violations=len(self._live_violations),
        )

    def _detect_regressions(self, run_result: RunResult) -> list[RegressionEntry]:
        if self.baseline is None:
            return []
        report = RegressionComparator().compare([self.baseline], [run_result])
        return report.entries
