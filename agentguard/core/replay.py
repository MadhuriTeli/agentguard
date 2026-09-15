"""Failure replay.

When a scenario fails, you want to reproduce it deterministically against a
patched agent — same inputs, same injected faults, same tool-call sequence
up to the point of divergence — without re-running the whole random chaos
schedule. `ReplaySession` captures that exact schedule from a Trajectory and
replays it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agentguard.core.runner import Runner, Scenario
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
    divergences: list[ReplayDivergence]

    @property
    def reproduced_failure(self) -> bool:
        """True if the replay still fails the same way the original did."""
        return (
            self.original_trajectory.success is False
            and self.replayed_trajectory.success is False
        )

    @property
    def fixed(self) -> bool:
        """True if a previously-failing trajectory now passes on replay."""
        return (
            self.original_trajectory.success is False
            and self.replayed_trajectory.success is True
        )


class DeterministicFaultSchedule:
    """A fault "engine" that replays a fixed, recorded sequence of faults
    instead of sampling randomly, so a replay run hits faults at exactly
    the same points as the original.
    """

    def __init__(self, recorded_faults: list[Optional[dict]]):
        self._faults = list(recorded_faults)
        self._index = 0

    def maybe_inject(self, observation: Any, trajectory: Trajectory) -> Any:
        if self._index >= len(self._faults):
            return observation
        fault = self._faults[self._index]
        self._index += 1
        if fault is None:
            return observation
        trajectory.add_step(StepType.FAULT_INJECTED, fault)
        return fault.get("mutated", observation)

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory) -> "DeterministicFaultSchedule":
        recorded = [
            step.content if step.type == StepType.FAULT_INJECTED else None
            for step in trajectory.steps
        ]
        return cls(recorded)


class ReplaySession:
    """Replays a previously recorded failing Trajectory against an agent."""

    def __init__(self, agent: Any):
        self.agent = agent

    def replay(self, original: Trajectory) -> ReplayResult:
        fault_schedule = DeterministicFaultSchedule.from_trajectory(original)
        runner = Runner(agent=self.agent, chaos_engine=fault_schedule)  # type: ignore[arg-type]

        first_message = next(
            (s.content for s in original.steps if s.type == StepType.MESSAGE), None
        )
        scenario = Scenario(
            name=f"replay::{original.scenario_name}",
            initial_input=first_message,
            max_steps=max(len(original.tool_calls()), 1) + 5,
        )

        replayed = runner.run(scenario)
        divergences = self._diff(original, replayed)

        return ReplayResult(
            original_trajectory=original,
            replayed_trajectory=replayed,
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
