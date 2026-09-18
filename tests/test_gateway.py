import pytest
from agentguard.tools.gateway import ContractViolation, ToolGateway, UnknownToolError

from agentguard.chaos.engine import ChaosEngine
from agentguard.chaos.faults import ToolErrorFault
from agentguard.contracts.schema import Contract, ToolConstraint
from agentguard.core.result import FailureType
from agentguard.core.trajectory import StepType, Trajectory


def _trajectory() -> Trajectory:
    return Trajectory(agent_name="a", scenario_name="s")


def test_has_tool_reflects_registered_tools():
    gateway = ToolGateway(tools={"reply": lambda **kw: kw})
    assert gateway.has_tool("reply")
    assert not gateway.has_tool("lookup_order")


def test_execute_unknown_tool_raises():
    gateway = ToolGateway(tools={"reply": lambda **kw: kw})
    with pytest.raises(UnknownToolError):
        gateway.execute("lookup_order", {}, _trajectory())


def test_execute_records_tool_result_step():
    gateway = ToolGateway(tools={"lookup_order": lambda order_id: {"status": "shipped"}})
    traj = _trajectory()

    result = gateway.execute("lookup_order", {"order_id": "1"}, traj)

    assert result == {"status": "shipped"}
    result_steps = [s for s in traj.steps if s.type == StepType.TOOL_RESULT]
    assert len(result_steps) == 1
    assert result_steps[0].content == {"status": "shipped"}


def test_execute_reply_also_records_assistant_message():
    gateway = ToolGateway(tools={"reply": lambda message: {"message": message}})
    traj = _trajectory()

    gateway.execute("reply", {"message": "here you go"}, traj)

    assistant_messages = [
        s
        for s in traj.steps
        if s.type == StepType.MESSAGE and s.metadata.get("role") == "assistant"
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == "here you go"


def test_execute_records_error_step_and_reraises_on_tool_exception():
    def broken(**kwargs):
        raise ValueError("boom")

    gateway = ToolGateway(tools={"broken": broken})
    traj = _trajectory()

    with pytest.raises(ValueError, match="boom"):
        gateway.execute("broken", {}, traj)

    error_steps = [s for s in traj.steps if s.type == StepType.ERROR]
    assert len(error_steps) == 1
    assert "boom" in error_steps[0].content


def test_forbidden_tool_raises_contract_violation_and_records_evidence():
    contract = Contract(name="c", forbidden_tools=["issue_refund"])
    gateway = ToolGateway(tools={"issue_refund": lambda **kw: kw}, contract=contract)
    traj = _trajectory()

    with pytest.raises(ContractViolation):
        gateway.execute("issue_refund", {"amount": 10}, traj)

    assert len(gateway.live_violations) == 1
    assert gateway.live_violations[0].failure_type == FailureType.FORBIDDEN_TOOL
    # the tool itself must never have actually run
    result_steps = [s for s in traj.steps if s.type == StepType.TOOL_RESULT]
    assert result_steps == []


def test_max_calls_exceeded_raises_on_the_call_that_breaches_it():
    contract = Contract(
        name="c", tool_constraints=[ToolConstraint(tool_name="lookup_order", max_calls=1)]
    )
    calls = []
    gateway = ToolGateway(
        tools={"lookup_order": lambda order_id: calls.append(order_id) or {"ok": True}},
        contract=contract,
    )
    traj = _trajectory()

    gateway.execute("lookup_order", {"order_id": "1"}, traj)  # first call: fine

    with pytest.raises(ContractViolation):
        gateway.execute("lookup_order", {"order_id": "1"}, traj)  # second: breaches

    assert calls == ["1"]  # second call never reached the real function
    assert gateway.live_violations[0].failure_type == FailureType.MAX_CALLS_EXCEEDED


def test_chaos_engine_mutates_the_result():
    chaos = ChaosEngine(faults=[ToolErrorFault(probability=1.0)])
    gateway = ToolGateway(tools={"flaky": lambda: {"status": "ok"}}, chaos_engine=chaos)
    traj = _trajectory()

    result = gateway.execute("flaky", {}, traj)

    assert result == {"error": "Simulated tool failure"}
    fault_steps = [s for s in traj.steps if s.type == StepType.FAULT_INJECTED]
    assert len(fault_steps) == 1


def test_call_counts_increment_per_execution():
    gateway = ToolGateway(tools={"reply": lambda **kw: kw})
    traj = _trajectory()

    gateway.execute("reply", {"message": "a"}, traj)
    gateway.execute("reply", {"message": "b"}, traj)

    assert gateway.tool_call_counts["reply"] == 2
