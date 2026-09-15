"""Runner: executes an agent against a scenario and records a Trajectory.

The Runner is intentionally agent-agnostic. It expects an `agent` object
exposing a `step(observation) -> action` style interface (or an async
equivalent), and a `scenario` describing the initial state and how to
produce observations / detect completion. Concrete agent adapters live
under `agents/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from agentguard.chaos.engine import ChaosEngine
from agentguard.core.trajectory import StepType, Trajectory


@dataclass
class Scenario:
    """Describes a single test case to run an agent through."""

    name: str
    initial_input: Any
    max_steps: int = 20
    is_done: Optional[Callable[[Trajectory], bool]] = None


class Runner:
    """Executes an agent through a scenario, optionally with chaos injection."""

    def __init__(self, agent: Any, chaos_engine: Optional[ChaosEngine] = None):
        self.agent = agent
        self.chaos_engine = chaos_engine

    def run(self, scenario: Scenario) -> Trajectory:
        trajectory = Trajectory(
            agent_name=getattr(self.agent, "name", self.agent.__class__.__name__),
            scenario_name=scenario.name,
        )

        observation = scenario.initial_input
        trajectory.add_step(StepType.MESSAGE, observation, role="user")

        try:
            for _ in range(scenario.max_steps):
                if self.chaos_engine is not None:
                    observation = self.chaos_engine.maybe_inject(observation, trajectory)

                action = self.agent.step(observation)
                trajectory.add_step(StepType.TOOL_CALL, action)

                observation = self._execute(action)
                trajectory.add_step(StepType.TOOL_RESULT, observation)

                if scenario.is_done and scenario.is_done(trajectory):
                    trajectory.finish(success=True)
                    return trajectory

            trajectory.finish(success=False)
        except Exception as exc:  # noqa: BLE001
            trajectory.add_step(StepType.ERROR, str(exc))
            trajectory.finish(success=False)

        return trajectory

    def _execute(self, action: Any) -> Any:
        """Execute an agent's chosen action/tool call and return the observation.

        Placeholder — wire this up to your actual tool execution layer.
        """
        raise NotImplementedError("Connect this to your agent's tool execution layer")
