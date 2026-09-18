"""ToolGateway: the single boundary every tool call passes through.

Before this, tool interception (call counting, contract enforcement, chaos
injection, trajectory recording) lived inline inside Harness as a
dict-of-wrapped-callables built once in __init__. That worked, but it
meant "the tool boundary" wasn't a thing you could point at, test in
isolation, or reuse outside Harness. ToolGateway makes it a first-class
object: every tool call goes through one fixed pipeline, in one place:

    1. Validate contract (forbidden tool / max_calls, live)
    2. Record the invocation (call counting)
    3. Execute the underlying tool
    4. Inject chaos faults into the result
    5. Record the result (and, for `reply`, the assistant message)

A ToolGateway is scoped to a single Harness.run() call — Harness builds a
fresh one per run so call counts and live-violation evidence never leak
between runs of the same Harness instance.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from agentguard.chaos.engine import ChaosEngine
from agentguard.contracts.schema import Contract
from agentguard.core.result import Evidence, FailureType
from agentguard.core.trajectory import StepType, Trajectory


class ContractViolation(Exception):
    """Raised when a live tool call breaks a contract's real-time rules."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnknownToolError(Exception):
    """Raised when the agent requests a tool the gateway doesn't have."""


class ToolGateway:
    def __init__(
        self,
        tools: dict[str, Callable[..., Any]],
        contract: Contract | None = None,
        chaos_engine: ChaosEngine | None = None,
    ) -> None:
        self.tools = tools
        self.contract = contract
        self.chaos_engine = chaos_engine
        self.tool_call_counts: Counter[str] = Counter()
        self.live_violations: list[Evidence] = []

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def execute(self, tool_name: str, args: dict[str, Any], trajectory: Trajectory) -> Any:
        """Run one tool call through the full gateway pipeline.

        Raises UnknownToolError if tool_name isn't registered, or
        ContractViolation if a live contract rule blocks this call (in
        which case the underlying tool is never invoked at all).
        """
        if not self.has_tool(tool_name):
            raise UnknownToolError(tool_name)

        self.tool_call_counts[tool_name] += 1
        self._enforce_live(tool_name, trajectory)

        try:
            result = self.tools[tool_name](**args)
        except Exception as exc:
            trajectory.add_step(StepType.ERROR, f"{tool_name} raised: {exc}")
            raise

        if self.chaos_engine is not None:
            result = self.chaos_engine.maybe_inject(result, trajectory)

        trajectory.add_step(StepType.TOOL_RESULT, result, tool=tool_name)

        # A reply tool represents the agent's user-facing response. Record
        # it as an assistant MESSAGE so trajectory-level evaluators (e.g.
        # TrajectoryJudge's final_response_contains) can inspect it.
        if tool_name == "reply":
            message = result.get("message", "") if isinstance(result, dict) else str(result)
            trajectory.add_step(StepType.MESSAGE, message, role="assistant")

        return result

    def _enforce_live(self, tool_name: str, trajectory: Trajectory) -> None:
        """Block/record contract violations that are knowable at call time."""
        if self.contract is None:
            return

        if self.contract.forbidden_tools and tool_name in self.contract.forbidden_tools:
            msg = f"Forbidden tool called: {tool_name}"
            evidence = Evidence(
                failure_type=FailureType.FORBIDDEN_TOOL,
                message=msg,
                expected=f"{tool_name} never called",
                actual=f"{tool_name} called",
            )
            self.live_violations.append(evidence)
            trajectory.add_step(StepType.ERROR, msg, kind="contract_violation")
            raise ContractViolation(msg)

        for constraint in self.contract.tool_constraints:
            if constraint.tool_name != tool_name or constraint.max_calls is None:
                continue
            if self.tool_call_counts[tool_name] > constraint.max_calls:
                count = self.tool_call_counts[tool_name]
                msg = f"Tool '{tool_name}' exceeded max_calls ({count} > {constraint.max_calls})"
                evidence = Evidence(
                    failure_type=FailureType.MAX_CALLS_EXCEEDED,
                    message=msg,
                    expected=f"<= {constraint.max_calls} calls",
                    actual=f"{count} calls",
                )
                self.live_violations.append(evidence)
                trajectory.add_step(StepType.ERROR, msg, kind="contract_violation")
                raise ContractViolation(msg)
