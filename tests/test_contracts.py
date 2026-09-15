from agentguard.contracts.schema import Contract, ToolConstraint
from agentguard.contracts.validator import ContractValidator
from agentguard.core.result import Verdict
from agentguard.core.trajectory import StepType, Trajectory


def make_trajectory_with_tool_calls(tool_names: list[str]) -> Trajectory:
    traj = Trajectory()
    for name in tool_names:
        traj.add_step(StepType.TOOL_CALL, {"tool": name})
    return traj


def test_passes_when_only_allowed_tools_used():
    contract = Contract(name="c1", allowed_tools=["lookup_order", "reply"])
    traj = make_trajectory_with_tool_calls(["lookup_order", "reply"])

    result = ContractValidator().validate(traj, contract)

    assert result.verdict == Verdict.PASS


def test_fails_when_forbidden_tool_used():
    contract = Contract(name="c2", forbidden_tools=["issue_refund"])
    traj = make_trajectory_with_tool_calls(["issue_refund"])

    result = ContractValidator().validate(traj, contract)

    assert result.verdict == Verdict.FAIL
    assert "issue_refund" in result.reason


def test_fails_when_max_calls_exceeded():
    contract = Contract(
        name="c3",
        tool_constraints=[ToolConstraint(tool_name="lookup_order", max_calls=1)],
    )
    traj = make_trajectory_with_tool_calls(["lookup_order", "lookup_order"])

    result = ContractValidator().validate(traj, contract)

    assert result.verdict == Verdict.FAIL
