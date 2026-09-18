"""The framework-agnostic boundary between Harness and any agent.

Before this, Harness called `agent.step(observation)` directly, which
means AgentGuard's entire value (contract enforcement, chaos injection,
trajectory tracing, replay) was only available to agents that happened to
expose exactly that method shape. AgentAdapter is the seam that fixes
that: Harness only ever talks to an AgentAdapter, never to a raw agent
object, so supporting a new agent framework (LangGraph, a raw OpenAI/
Anthropic tool-use loop, anything) means writing one adapter class rather
than changing Harness.

This module deliberately does NOT ship adapters for every framework —
per the review that led here, the point right now is establishing the
abstraction, not building out every integration. NativeAgentAdapter (in
native.py) is the only concrete adapter that exists today, wrapping
AgentGuard's own pre-existing `step(observation)` interface so nothing
that already worked breaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agentguard.core.trajectory import Trajectory


@dataclass
class ExecutionContext:
    """Context available to an adapter when deciding the next action.

    Carries the trajectory recorded so far (so an adapter for a framework
    with richer state than "the last observation" can use it) and the
    step number within the current run. Native agents can ignore this
    entirely, as NativeAgentAdapter does.
    """

    trajectory: Trajectory
    step_number: int


@dataclass
class AgentExecution:
    """One decision an agent made, in the shape Harness already executes.

    `action` keeps the existing {"tool": ..., "args": {...}} shape so
    adding the adapter layer requires no change to how Harness executes
    the chosen tool — only to how the decision itself gets obtained.
    """

    action: dict[str, Any]


@runtime_checkable
class AgentAdapter(Protocol):
    """The only interface Harness depends on to get an agent's next move.

    @runtime_checkable makes `isinstance(x, AgentAdapter)` check for the
    presence of `name` and `run`, so Harness can tell an adapter apart
    from a raw native-style agent (which only has `step`) and wrap the
    latter automatically — see Harness.__init__.
    """

    @property
    def name(self) -> str: ...

    def run(self, observation: Any, context: ExecutionContext) -> AgentExecution: ...
