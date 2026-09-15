"""Chaos engine: injects faults into an agent run to test resilience."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from agentguard.chaos.faults import Fault
from agentguard.core.trajectory import StepType, Trajectory


@dataclass
class ChaosEngine:
    """Applies configured faults probabilistically during a run."""

    faults: list[Fault] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def maybe_inject(self, observation: Any, trajectory: Trajectory) -> Any:
        """Possibly mutate `observation` by applying an eligible fault."""
        for fault in self.faults:
            if self._rng.random() < fault.probability:
                mutated = fault.apply(observation)
                trajectory.add_step(
                    StepType.FAULT_INJECTED,
                    {"fault": fault.name, "original": observation, "mutated": mutated},
                )
                observation = mutated
        return observation
