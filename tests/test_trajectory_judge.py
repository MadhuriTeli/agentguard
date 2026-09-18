from agentguard.core.result import FailureType, Verdict
from agentguard.core.trajectory import StepType, Trajectory
from agentguard.judges.trajectory_judge import TrajectoryJudge


def _trajectory_with(tool_calls: list[dict], final_reply: str | None = None) -> Trajectory:
    traj = Trajectory(agent_name="a", scenario_name="s")
    for call in tool_calls:
        traj.add_step(StepType.TOOL_CALL, call)
    if final_reply is not None:
        traj.add_step(StepType.MESSAGE, final_reply, role="assistant")
    return traj


def test_passes_when_no_expectations_set():
    judge = TrajectoryJudge()
    result = judge.evaluate(_trajectory_with([]))

    assert result.verdict == Verdict.PASS


def test_passes_when_tool_calls_match_exactly():
    judge = TrajectoryJudge(
        expected_tool_calls=[
            {"tool": "lookup_order", "args": {"order_id": "123"}},
            {"tool": "reply", "args": {}},
        ]
    )
    traj = _trajectory_with(
        [
            {"tool": "lookup_order", "args": {"order_id": "123"}},
            {"tool": "reply", "args": {"message": "here you go"}},
        ]
    )

    result = judge.evaluate(traj)

    assert result.verdict == Verdict.PASS


def test_fails_when_tool_name_differs_at_a_position():
    judge = TrajectoryJudge(expected_tool_calls=[{"tool": "verify_identity", "args": {}}])
    traj = _trajectory_with([{"tool": "issue_refund", "args": {"amount": 100}}])

    result = judge.evaluate(traj)

    assert result.verdict == Verdict.FAIL
    assert "verify_identity" in result.reason
    assert "issue_refund" in result.reason


def test_fails_when_expected_arg_value_differs():
    judge = TrajectoryJudge(
        expected_tool_calls=[{"tool": "lookup_order", "args": {"order_id": "123"}}]
    )
    traj = _trajectory_with([{"tool": "lookup_order", "args": {"order_id": "999"}}])

    result = judge.evaluate(traj)

    assert result.verdict == Verdict.FAIL
    assert "order_id" in result.reason


def test_fails_when_fewer_tool_calls_than_expected():
    judge = TrajectoryJudge(
        expected_tool_calls=[
            {"tool": "confirm_with_user", "args": {}},
            {"tool": "cancel_subscription", "args": {}},
        ]
    )
    traj = _trajectory_with([{"tool": "confirm_with_user", "args": {}}])

    result = judge.evaluate(traj)

    assert result.verdict == Verdict.FAIL
    assert "at least 2" in result.reason


def test_extra_trailing_tool_calls_beyond_expected_are_not_a_failure():
    """expected_tool_calls checks a prefix, not an exact-length match — an
    agent that does the required calls plus something extra afterward
    (e.g. a final reply) shouldn't fail on that account alone.
    """
    judge = TrajectoryJudge(expected_tool_calls=[{"tool": "lookup_policy", "args": {}}])
    traj = _trajectory_with(
        [{"tool": "lookup_policy", "args": {"topic": "returns"}}, {"tool": "reply", "args": {}}]
    )

    result = judge.evaluate(traj)

    assert result.verdict == Verdict.PASS


def test_passes_when_final_response_contains_expected_text():
    judge = TrajectoryJudge(final_response_contains=["30 days"])
    traj = _trajectory_with([], final_reply="Our return policy allows 30 days.")

    result = judge.evaluate(traj)

    assert result.verdict == Verdict.PASS


def test_final_response_check_is_case_insensitive():
    judge = TrajectoryJudge(final_response_contains=["CAN'T ISSUE A REFUND"])
    traj = _trajectory_with([], final_reply="I can't issue a refund without verification.")

    result = judge.evaluate(traj)

    assert result.verdict == Verdict.PASS


def test_fails_when_final_response_missing_expected_text():
    judge = TrajectoryJudge(final_response_contains=["30 days"])
    traj = _trajectory_with([], final_reply="Your request has been processed.")

    result = judge.evaluate(traj)

    assert result.verdict == Verdict.FAIL
    assert "30 days" in result.reason


def test_fails_when_no_assistant_message_recorded_at_all():
    """If nothing ever tagged a MESSAGE step with role=assistant (e.g. the
    Harness's reply-interception logic regresses), this must fail loudly
    rather than silently treating an empty string as 'no response expected'.
    """
    judge = TrajectoryJudge(final_response_contains=["anything"])
    traj = _trajectory_with([])  # no final_reply at all

    result = judge.evaluate(traj)

    assert result.verdict == Verdict.FAIL


def test_uses_the_last_assistant_message_when_multiple_exist():
    judge = TrajectoryJudge(final_response_contains=["final answer"])
    traj = _trajectory_with([])
    traj.add_step(StepType.MESSAGE, "thinking out loud", role="assistant")
    traj.add_step(StepType.MESSAGE, "here is the final answer", role="assistant")

    result = judge.evaluate(traj)

    assert result.verdict == Verdict.PASS


def test_ignores_user_messages_when_finding_final_response():
    judge = TrajectoryJudge(final_response_contains=["30 days"])
    traj = Trajectory(agent_name="a", scenario_name="s")
    traj.add_step(StepType.MESSAGE, "does this contain 30 days?", role="user")

    result = judge.evaluate(traj)

    assert result.verdict == Verdict.FAIL


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def test_wrong_tool_at_position_is_classified_as_wrong_tool():
    judge = TrajectoryJudge(expected_tool_calls=[{"tool": "verify_identity", "args": {}}])
    traj = _trajectory_with([{"tool": "issue_refund", "args": {"amount": 100}}])

    result = judge.evaluate(traj)

    assert result.failure_types == [FailureType.WRONG_TOOL]
    assert result.evidence[0].expected == "verify_identity"
    assert result.evidence[0].actual == "issue_refund"


def test_wrong_argument_is_classified_as_wrong_argument_not_wrong_tool():
    judge = TrajectoryJudge(
        expected_tool_calls=[{"tool": "lookup_order", "args": {"order_id": "123"}}]
    )
    traj = _trajectory_with([{"tool": "lookup_order", "args": {"order_id": "999"}}])

    result = judge.evaluate(traj)

    assert result.failure_types == [FailureType.WRONG_ARGUMENT]


def test_too_few_tool_calls_is_classified_as_premature_termination():
    judge = TrajectoryJudge(
        expected_tool_calls=[
            {"tool": "confirm_with_user", "args": {}},
            {"tool": "cancel_subscription", "args": {}},
        ]
    )
    traj = _trajectory_with([{"tool": "confirm_with_user", "args": {}}])

    result = judge.evaluate(traj)

    assert FailureType.PREMATURE_TERMINATION in result.failure_types


def test_missing_final_response_text_with_no_reply_at_all_is_incomplete_response():
    """No assistant message exists at all — that's a different fact than
    'the agent replied, but said the wrong thing'.
    """
    judge = TrajectoryJudge(final_response_contains=["30 days"])
    traj = _trajectory_with([])  # no final_reply

    result = judge.evaluate(traj)

    assert result.failure_types == [FailureType.INCOMPLETE_RESPONSE]


def test_wrong_final_response_text_with_a_reply_present_is_wrong_final_answer():
    judge = TrajectoryJudge(final_response_contains=["30 days"])
    traj = _trajectory_with([], final_reply="Your request has been processed.")

    result = judge.evaluate(traj)

    assert result.failure_types == [FailureType.WRONG_FINAL_ANSWER]
