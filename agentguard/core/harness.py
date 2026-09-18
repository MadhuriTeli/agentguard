"""The AgentGuard Harness.

This is the single entry point that ties every other piece together:

    Agent
      |
    Harness
      |-- AgentAdapter          (framework-agnostic: wraps native step()
      |                          agents automatically; anything else
      |                          implements AgentAdapter directly)
      |-- ToolGateway           (contract enforcement, chaos injection,
      |                          and trajectory recording all happen here,
      |                          as one fixed pipeline every tool call
      |                          passes through — see agentguard.tools.gateway)
      |-- evaluate result       (contract validator + judges)
      |-- calculate metrics     (latency, call counts, fault/violation counts)
      \\-- detect regressions    (vs an optional baseline RunResult)
      |
    PASS / FAIL

Where the earlier pieces (Runner, ContractValidator, ChaosEngine,
EvaluationPipeline, RegressionComparator, ToolGateway) are independent
building blocks you can use standalone, the Harness is the opinionated
orchestration layer most users will actually reach for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentguard.adapters.base import AgentAdapter, ExecutionContext
from agentguard.adapters.native import NativeAgentAdapter
from agentguard.chaos.engine import ChaosEngine
from agentguard.contracts.schema import Contract
from agentguard.contracts.validator import ContractValidator
from agentguard.core.evaluator import EvaluationPipeline, Evaluator
from agentguard.core.result import EvalResult, RunResult, Verdict
from agentguard.core.runner import Scenario
from agentguard.core.trajectory import StepType, Trajectory
from agentguard.regression.comparator import RegressionComparator, RegressionEntry
from agentguard.tools.gateway import ContractViolation, ToolGateway, UnknownToolError

__all__ = [
    "ContractViolation",
    "Harness",
    "HarnessMetrics",
    "HarnessResult",
]


@dataclass
class HarnessMetrics:
    """Quantitative summary of a single harness run."""

    duration_seconds: float | None
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
    contract_result: EvalResult | None
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
                {
                    "verdict": self.contract_result.verdict.value,
                    "reason": self.contract_result.reason,
                }
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
        contract: Contract | None = None,
        chaos_engine: ChaosEngine | None = None,
        evaluators: list[Evaluator] | None = None,
        baseline: RunResult | None = None,
    ):
        # Harness never talks to a raw agent object — only to an
        # AgentAdapter. Anything that isn't already one (i.e. every
        # existing native step()-style agent) gets auto-wrapped so nothing
        # that worked before this abstraction existed needs to change.
        self.adapter: AgentAdapter = (
            agent if isinstance(agent, AgentAdapter) else NativeAgentAdapter(agent)
        )
        self.raw_tools = tools
        self.contract = contract
        self.chaos_engine = chaos_engine
        self.evaluation_pipeline = EvaluationPipeline(evaluators or [])
        self.baseline = baseline

        # Built fresh per run() call — see run() below — so call counts
        # and live-violation evidence never leak between runs of the same
        # Harness instance.
        self.gateway: ToolGateway | None = None

    # -- orchestration ------------------------------------------------------

    def run(self, scenario: Scenario) -> HarnessResult:
        gateway = ToolGateway(self.raw_tools, self.contract, self.chaos_engine)
        self.gateway = gateway

        trajectory = Trajectory(
            agent_name=self.adapter.name,
            scenario_name=scenario.name,
            initial_input=scenario.initial_input,
        )

        observation: Any = scenario.initial_input
        trajectory.add_step(StepType.MESSAGE, observation, role="user")

        success = False
        try:
            for step_number in range(scenario.max_steps):
                context = ExecutionContext(trajectory=trajectory, step_number=step_number)
                execution = self.adapter.run(observation, context)
                action = execution.action
                trajectory.add_step(StepType.TOOL_CALL, action)

                tool_name = action.get("tool") if isinstance(action, dict) else None
                args = action.get("args", {}) if isinstance(action, dict) else {}

                if tool_name is None or not gateway.has_tool(tool_name):
                    trajectory.add_step(StepType.ERROR, f"Unknown tool requested: {tool_name}")
                    break

                observation = gateway.execute(tool_name, args, trajectory)

                if scenario.is_done and scenario.is_done(trajectory):
                    success = True
                    break
            else:
                success = False
        except ContractViolation:
            success = False
        except UnknownToolError as exc:
            trajectory.add_step(StepType.ERROR, f"Unknown tool requested: {exc}")
            success = False
        except Exception as exc:  # noqa: BLE001
            trajectory.add_step(StepType.ERROR, str(exc))
            success = False

        trajectory.finish(success=success and not gateway.live_violations)

        contract_result = self._evaluate_contract(trajectory, gateway)
        run_result = self.evaluation_pipeline.run(trajectory, self.contract)
        metrics = self._calculate_metrics(trajectory, gateway)
        regression_entries = self._detect_regressions(run_result)

        return HarnessResult(
            trajectory=trajectory,
            run_result=run_result,
            contract_result=contract_result,
            metrics=metrics,
            regression_entries=regression_entries,
        )

    # -- post-run steps -----------------------------------------------------

    def _evaluate_contract(self, trajectory: Trajectory, gateway: ToolGateway) -> EvalResult | None:
        if self.contract is None:
            return None

        result = ContractValidator().validate(trajectory, self.contract)

        if gateway.live_violations:
            live_messages = [e.message for e in gateway.live_violations]
            result = EvalResult(
                name=result.name,
                verdict=Verdict.FAIL,
                reason="; ".join([*live_messages, result.reason]),
                details={
                    "live_violations": live_messages,
                    **result.details,
                },
                evidence=[*gateway.live_violations, *result.evidence],
            )
        return result

    def _calculate_metrics(self, trajectory: Trajectory, gateway: ToolGateway) -> HarnessMetrics:
        return HarnessMetrics(
            duration_seconds=trajectory.duration_seconds,
            num_steps=len(trajectory.steps),
            num_tool_calls=sum(gateway.tool_call_counts.values()),
            tool_call_counts=dict(gateway.tool_call_counts),
            num_faults_injected=sum(
                1 for s in trajectory.steps if s.type == StepType.FAULT_INJECTED
            ),
            num_live_violations=len(gateway.live_violations),
        )

    def _detect_regressions(self, run_result: RunResult) -> list[RegressionEntry]:
        if self.baseline is None:
            return []
        report = RegressionComparator().compare([self.baseline], [run_result])
        return report.entries
