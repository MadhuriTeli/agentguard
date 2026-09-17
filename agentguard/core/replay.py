"""Failure replay.

When a scenario fails, you want to reproduce it deterministically against a
patched agent — same inputs, same injected faults, same tool-call sequence
up to the point of divergence — without re-running the whole random chaos
schedule. `ReplaySession` captures that exact schedule from a Trajectory and
replays it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentguard.chaos.engine import ChaosEngine
from agentguard.contracts.schema import Contract
from agentguard.core.evaluator import Evaluator
from agentguard.core.harness import Harness, HarnessResult
from agentguard.core.runner import Scenario
from agentguard.core.trajectory import StepType, Trajectory


@dataclass
class ReplayDivergence:
    """Where the replayed run's behavior differed from the original."""

    step_index: int
    original: Any
    replayed: Any


@dataclass
class ReplayResult:
    original_trajectory: Trajectory
    replayed_trajectory: Trajectory
    original_passed: bool
    replayed_passed: bool
    divergences: list[ReplayDivergence]

    @property
    def reproduced_failure(self) -> bool:
        """True if the replay still fails the way the original did.

        Deliberately keyed off the ORIGINAL RUN'S evaluator/contract
        verdict (HarnessResult.passed), not Trajectory.success. Those are
        different things: success reflects whether the scenario's
        `is_done` condition fired (did the loop reach a stopping point),
        which can be True even for a run that completed but violated its
        contract or failed a judge. Replay verification needs to answer
        "did this fix the thing that made evaluation fail", so it must
        compare verdicts, not loop-completion state.
        """
        return not self.original_passed and not self.replayed_passed

    @property
    def fixed(self) -> bool:
        """True if a previously-failing run's evaluator/contract verdict
        now passes on replay. See reproduced_failure's docstring for why
        this is keyed off verdicts rather than Trajectory.success.
        """
        return not self.original_passed and self.replayed_passed


@dataclass
class FaultBatch:
    """All faults injected during a single tool execution.

    There can be zero, one, or several — ChaosEngine.maybe_inject loops
    over every configured Fault and applies each independently, so one
    tool call can trigger multiple faults in sequence.
    """

    faults: list[dict[str, Any]] = field(default_factory=list)


class DeterministicFaultSchedule(ChaosEngine):
    """A fault "engine" that replays a fixed, recorded sequence of fault
    BATCHES — one batch per tool execution — instead of sampling randomly,
    so a replay run hits the same faults on the same calls as the original.

    This is keyed by TOOL-EXECUTION index, deliberately not by
    trajectory-STEP index. Those are not the same sequence: each tool
    execution produces one TOOL_CALL step, zero or more FAULT_INJECTED
    steps, and one TOOL_RESULT step, while Harness calls `maybe_inject`
    exactly once per execution. An earlier version of this class indexed
    directly into `trajectory.steps`, which silently misaligns after the
    very first tool call that injects zero or more-than-one faults, since
    the two sequences advance at different rates — a fault recorded at
    step index 4 might need to fire on tool-execution index 1, not 4.
    """

    def __init__(self, batches: list[FaultBatch]):
        self._batches = list(batches)
        self._index = 0

    def maybe_inject(self, observation: Any, trajectory: Trajectory) -> Any:
        batch = self._batches[self._index] if self._index < len(self._batches) else FaultBatch()
        self._index += 1

        for fault in batch.faults:
            trajectory.add_step(StepType.FAULT_INJECTED, fault)
            observation = fault.get("mutated", observation)

        return observation

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory) -> DeterministicFaultSchedule:
        """Group each tool execution's faults into one FaultBatch, in the
        order the executions happened.

        A tool execution's faults are exactly the FAULT_INJECTED steps
        recorded between the previous TOOL_RESULT (or the start of the
        trajectory) and this execution's own TOOL_RESULT — because Harness
        only calls `maybe_inject` once per successful tool call,
        immediately before recording that call's TOOL_RESULT. A TOOL_CALL
        that raises before reaching `maybe_inject` produces an ERROR step
        instead of a TOOL_RESULT and correctly gets no batch at all, since
        `maybe_inject` was never invoked for it.
        """
        batches: list[FaultBatch] = []
        pending: list[dict[str, Any]] = []

        for step in trajectory.steps:
            if step.type == StepType.FAULT_INJECTED:
                pending.append(step.content)
            elif step.type == StepType.TOOL_RESULT:
                batches.append(FaultBatch(faults=pending))
                pending = []
            # MESSAGE / TOOL_CALL / ERROR steps don't close a batch.

        return cls(batches)


class ReplaySession:
    """Replays a previously recorded failing run against an agent.

    Takes the ORIGINAL run's full HarnessResult, not just its Trajectory —
    replay verification needs the original's pass/fail verdict to compare
    against (see ReplayResult.fixed's docstring), and the trajectory alone
    doesn't carry that.
    """

    def __init__(
        self,
        agent: Any,
        tools: dict[str, Callable[..., Any]],
        contract: Contract | None = None,
        evaluators: list[Evaluator] | None = None,
    ) -> None:
        self.agent = agent
        self.tools = tools
        self.contract = contract
        self.evaluators = evaluators or []

    def replay(self, original_run: HarnessResult) -> ReplayResult:
        original = original_run.trajectory
        fault_schedule = DeterministicFaultSchedule.from_trajectory(original)

        harness = Harness(
            agent=self.agent,
            tools=self.tools,
            contract=self.contract,
            chaos_engine=fault_schedule,
            evaluators=self.evaluators,
        )

        scenario = Scenario(
            name=f"replay::{original.scenario_name}",
            initial_input=original.initial_input,
            max_steps=max(len(original.tool_calls()), 1) + 5,
        )

        replayed_run = harness.run(scenario)
        divergences = self._diff(original, replayed_run.trajectory)

        return ReplayResult(
            original_trajectory=original,
            replayed_trajectory=replayed_run.trajectory,
            original_passed=original_run.passed,
            replayed_passed=replayed_run.passed,
            divergences=divergences,
        )

    @staticmethod
    def _diff(original: Trajectory, replayed: Trajectory) -> list[ReplayDivergence]:
        divergences = []
        for i, (orig_step, replay_step) in enumerate(zip(original.steps, replayed.steps)):
            if orig_step.type != replay_step.type or orig_step.content != replay_step.content:
                divergences.append(
                    ReplayDivergence(
                        step_index=i,
                        original=orig_step.content,
                        replayed=replay_step.content,
                    )
                )
        return divergences
