from agentguard.contracts.schema import Contract, ToolConstraint
from agentguard.contracts.validator import ContractValidator
from agentguard.core.result import FailureType, Verdict
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


# ---------------------------------------------------------------------------
# Failure classification: evidence and failure_type, not just pass/fail
# ---------------------------------------------------------------------------


def test_forbidden_tool_produces_forbidden_tool_evidence():
    contract = Contract(name="c10", forbidden_tools=["issue_refund"])
    traj = make_trajectory_with_tool_calls(["issue_refund"])

    result = ContractValidator().validate(traj, contract)

    assert FailureType.FORBIDDEN_TOOL in result.failure_types
    ev = result.evidence[0]
    assert ev.actual == "issue_refund"


def test_max_calls_exceeded_produces_max_calls_exceeded_evidence():
    contract = Contract(
        name="c11",
        tool_constraints=[ToolConstraint(tool_name="lookup_order", max_calls=1)],
    )
    traj = make_trajectory_with_tool_calls(["lookup_order", "lookup_order"])

    result = ContractValidator().validate(traj, contract)

    assert result.failure_types == [FailureType.MAX_CALLS_EXCEEDED]
    assert result.evidence[0].expected == "<= 1 calls"
    assert result.evidence[0].actual == "2 calls"


def test_required_step_never_called_is_missing_not_ordering():
    """A step that never appears at all is a different, more basic fact
    than a step that appears too late — MISSING_REQUIRED_STEP, not
    ORDERING_VIOLATION.
    """
    contract = Contract(name="c12", required_steps=["verify_identity"])
    traj = make_trajectory_with_tool_calls(["lookup_order", "reply"])

    result = ContractValidator().validate(traj, contract)

    assert result.failure_types == [FailureType.MISSING_REQUIRED_STEP]


def test_required_step_called_out_of_order_is_ordering_not_missing():
    """verify_identity DID happen, just after issue_refund rather than
    before it — this is an ordering fact, not an absence fact, and a
    developer debugging this needs to know which.
    """
    contract = Contract(name="c13", required_steps=["verify_identity", "issue_refund"])
    traj = make_trajectory_with_tool_calls(["issue_refund", "verify_identity"])

    result = ContractValidator().validate(traj, contract)

    assert result.failure_types == [FailureType.ORDERING_VIOLATION]


def test_max_turns_exceeded_produces_matching_evidence():
    contract = Contract(name="c14", max_turns=1)
    traj = Trajectory(agent_name="a", scenario_name="s")
    traj.add_step(StepType.MESSAGE, "hi")
    traj.add_step(StepType.MESSAGE, "there")

    result = ContractValidator().validate(traj, contract)

    assert result.failure_types == [FailureType.MAX_TURNS_EXCEEDED]


def test_passing_contract_has_no_evidence():
    contract = Contract(name="c15", allowed_tools=["reply"])
    traj = make_trajectory_with_tool_calls(["reply"])

    result = ContractValidator().validate(traj, contract)

    assert result.evidence == []
    assert result.primary_failure_type is None


def test_multiple_simultaneous_violations_each_get_their_own_evidence():
    contract = Contract(
        name="c16",
        forbidden_tools=["issue_refund"],
        required_steps=["verify_identity"],
    )
    traj = make_trajectory_with_tool_calls(["issue_refund"])

    result = ContractValidator().validate(traj, contract)

    assert set(result.failure_types) == {
        FailureType.FORBIDDEN_TOOL,
        FailureType.MISSING_REQUIRED_STEP,
    }
    assert len(result.evidence) == 2
