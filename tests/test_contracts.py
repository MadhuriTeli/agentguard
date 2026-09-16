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


# ---------------------------------------------------------------------------
# required_steps: an ORDERED sequence (see ContractValidator's cursor-based
# subsequence match). These cases exist because the ordering behavior has
# real, easy-to-get-wrong edge cases that "was it called at all" tests
# wouldn't catch.
# ---------------------------------------------------------------------------


def test_required_steps_passes_when_called_in_order():
    contract = Contract(name="c4", required_steps=["verify_identity", "issue_refund"])
    traj = make_trajectory_with_tool_calls(["verify_identity", "issue_refund"])

    result = ContractValidator().validate(traj, contract)

    assert result.verdict == Verdict.PASS


def test_required_steps_passes_with_extra_calls_interspersed():
    """Unrelated calls before/between/after required steps shouldn't matter —
    only that the required ones occur, in order, as a subsequence."""
    contract = Contract(name="c5", required_steps=["verify_identity", "issue_refund"])
    traj = make_trajectory_with_tool_calls(
        ["lookup_policy", "verify_identity", "reply", "issue_refund", "reply"]
    )

    result = ContractValidator().validate(traj, contract)

    assert result.verdict == Verdict.PASS


def test_required_steps_fails_when_never_called():
    contract = Contract(name="c6", required_steps=["verify_identity"])
    traj = make_trajectory_with_tool_calls(["lookup_order", "reply"])

    result = ContractValidator().validate(traj, contract)

    assert result.verdict == Verdict.FAIL
    assert "verify_identity" in result.reason


def test_required_steps_fails_when_called_out_of_order():
    """verify_identity happening AFTER issue_refund does not satisfy a
    contract requiring it to happen BEFORE — order is semantically load-
    bearing here (you can't retroactively verify a refund that already
    went out), so a naive "were both tools called" check would wrongly
    pass this.
    """
    contract = Contract(name="c7", required_steps=["verify_identity", "issue_refund"])
    traj = make_trajectory_with_tool_calls(["issue_refund", "verify_identity"])

    result = ContractValidator().validate(traj, contract)

    assert result.verdict == Verdict.FAIL
    # verify_identity WAS eventually called, so only the still-missing
    # "issue_refund after that point" should be reported.
    assert "issue_refund" in result.reason
    assert "verify_identity" not in result.reason.split("issue_refund")[0]


def test_required_steps_fails_when_only_partially_satisfied_in_order():
    contract = Contract(name="c8", required_steps=["confirm_with_user", "cancel_subscription"])
    traj = make_trajectory_with_tool_calls(["confirm_with_user", "reply"])

    result = ContractValidator().validate(traj, contract)

    assert result.verdict == Verdict.FAIL
    assert "cancel_subscription" in result.reason
    assert "confirm_with_user" not in result.reason


def test_empty_required_steps_always_passes():
    contract = Contract(name="c9", required_steps=[])
    traj = make_trajectory_with_tool_calls([])

    result = ContractValidator().validate(traj, contract)

    assert result.verdict == Verdict.PASS
