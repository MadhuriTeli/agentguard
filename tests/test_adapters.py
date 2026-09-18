"""Tests for the AgentAdapter boundary.

The point of AgentAdapter is that Harness works with ANY agent shape, not
just AgentGuard's own step()-style native agents. test_custom_adapter_*
below is the test that actually proves that — everything else here proves
backward compatibility (existing native agents keep working unmodified).
"""

from __future__ import annotations

from typing import Any

from agentguard.adapters.base import AgentAdapter, AgentExecution, ExecutionContext
from agentguard.adapters.native import NativeAgentAdapter

from agentguard.core.harness import Harness
from agentguard.core.runner import Scenario


class _NativeStyleAgent:
    """An ordinary step()-style agent, same shape as CustomerSupportAgent."""

    name = "native_agent"

    def __init__(self):
        self.calls = 0

    def step(self, observation: Any) -> dict:
        self.calls += 1
        return {"tool": "reply", "args": {"message": f"call-{self.calls}"}}


class _CustomFrameworkAgent:
    """Deliberately NOT step()-shaped — simulates a different agent
    framework's native interface, to prove Harness can run it directly
    once wrapped in an AgentAdapter, without any NativeAgentAdapter
    involvement at all.
    """

    def decide_next_move(self, last_result: Any) -> dict:
        return {"tool": "reply", "args": {"message": "handled by a custom framework"}}


class _CustomFrameworkAdapter:
    """A genuinely custom AgentAdapter — not a subclass of anything
    AgentGuard ships, just something that satisfies the Protocol
    structurally. This is the actual test of framework-agnosticism.
    """

    def __init__(self, framework_agent: _CustomFrameworkAgent):
        self._framework_agent = framework_agent
        self.contexts_seen: list[ExecutionContext] = []

    @property
    def name(self) -> str:
        return "custom_framework_adapter"

    def run(self, observation: Any, context: ExecutionContext) -> AgentExecution:
        self.contexts_seen.append(context)
        action = self._framework_agent.decide_next_move(observation)
        return AgentExecution(action=action)


def _done_after_n_tool_calls(n: int):
    def is_done(trajectory):
        return len(trajectory.tool_calls()) >= n

    return is_done


def test_native_agent_adapter_wraps_step_style_agent():
    agent = _NativeStyleAgent()
    adapter = NativeAgentAdapter(agent)

    assert adapter.name == "native_agent"

    context = ExecutionContext(trajectory=None, step_number=0)  # type: ignore[arg-type]
    execution = adapter.run("hello", context)

    assert execution.action == {"tool": "reply", "args": {"message": "call-1"}}


def test_native_agent_adapter_falls_back_to_class_name_without_name_attr():
    class _Anonymous:
        def step(self, observation):
            return {"tool": "reply", "args": {}}

    adapter = NativeAgentAdapter(_Anonymous())
    assert adapter.name == "_Anonymous"


def test_harness_auto_wraps_a_plain_native_agent():
    tools = {"reply": lambda message: {"message": message}}
    agent = _NativeStyleAgent()
    harness = Harness(agent=agent, tools=tools)

    assert not isinstance(agent, AgentAdapter)  # sanity: it's genuinely native
    assert isinstance(harness.adapter, NativeAgentAdapter)

    result = harness.run(
        Scenario(name="s1", initial_input="hi", is_done=_done_after_n_tool_calls(1))
    )
    assert result.passed
    assert result.trajectory.agent_name == "native_agent"


def test_harness_uses_a_custom_adapter_directly_without_double_wrapping():
    """The actual proof of framework-agnosticism: an agent shape AgentGuard
    has never seen (decide_next_move, not step) runs correctly through
    Harness purely because its adapter satisfies AgentAdapter.
    """
    tools = {"reply": lambda message: {"message": message}}
    custom_adapter = _CustomFrameworkAdapter(_CustomFrameworkAgent())
    harness = Harness(agent=custom_adapter, tools=tools)

    assert isinstance(custom_adapter, AgentAdapter)
    # Harness must use the adapter AS GIVEN, not wrap it a second time.
    assert harness.adapter is custom_adapter

    result = harness.run(
        Scenario(name="s2", initial_input="hi", is_done=_done_after_n_tool_calls(1))
    )

    assert result.passed
    assert result.trajectory.agent_name == "custom_framework_adapter"
    assert "handled by a custom framework" in str(result.trajectory.steps[-1].content)


def test_execution_context_carries_trajectory_and_step_number():
    tools = {"reply": lambda message: {"message": message}}
    custom_adapter = _CustomFrameworkAdapter(_CustomFrameworkAgent())
    harness = Harness(agent=custom_adapter, tools=tools)

    harness.run(Scenario(name="s3", initial_input="hi", is_done=_done_after_n_tool_calls(1)))

    assert len(custom_adapter.contexts_seen) == 1
    ctx = custom_adapter.contexts_seen[0]
    assert ctx.step_number == 0
    assert ctx.trajectory is not None
    assert ctx.trajectory.scenario_name == "s3"
