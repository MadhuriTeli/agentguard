"""Tests for failure replay.

Two tiers here:
1. Unit tests for DeterministicFaultSchedule, which lock in the exact
   tool-execution-vs-trajectory-step alignment bug found in review (see
   replay.py's docstrings for the full explanation).
2. A real end-to-end integration test using the actual CustomerSupportAgent,
   its real tools, chaos, and a contract-free evaluator — proving a broken
   agent's failure is faithfully reproduced, and that swapping in a patched
   agent against the SAME recorded run verifiably resolves it. This is the
   "flagship" scenario replay exists to support.
"""

from __future__ import annotations

from agentguard.chaos.engine import ChaosEngine
from agentguard.chaos.faults import MalformedResponseFault, ToolErrorFault
from agentguard.core.harness import Harness
from agentguard.core.replay import DeterministicFaultSchedule, FaultBatch, ReplaySession
from agentguard.core.runner import Scenario
from agentguard.core.trajectory import StepType, Trajectory
from agentguard.judges.trajectory_judge import TrajectoryJudge
from agents.customer_support.agent import CustomerSupportAgent
from agents.customer_support.tools import TOOLS

# ---------------------------------------------------------------------------
# Unit tests: DeterministicFaultSchedule's tool-execution-based alignment
# ---------------------------------------------------------------------------


def _trajectory_with_raw_steps(steps: list[tuple[StepType, object]]) -> Trajectory:
    traj = Trajectory(agent_name="a", scenario_name="s")
    for step_type, content in steps:
        traj.add_step(step_type, content)
    return traj


def test_single_fault_on_second_of_three_calls_replays_on_the_right_call():
    """This is the exact scenario that broke under step-index-based
    alignment: a fault recorded partway through the trajectory must fire
    on the SAME tool execution during replay, not an earlier or later one.
    """
    original = _trajectory_with_raw_steps(
        [
            (StepType.MESSAGE, "hi"),
            (StepType.TOOL_CALL, {"tool": "a"}),
            (StepType.TOOL_RESULT, {"ok": True}),  # execution 0: no fault
            (StepType.TOOL_CALL, {"tool": "b"}),
            (StepType.FAULT_INJECTED, {"fault": "tool_error", "mutated": {"error": "boom"}}),
            (StepType.TOOL_RESULT, {"error": "boom"}),  # execution 1: one fault
            (StepType.TOOL_CALL, {"tool": "reply"}),
            (StepType.TOOL_RESULT, {"message": "done"}),  # execution 2: no fault
        ]
    )

    schedule = DeterministicFaultSchedule.from_trajectory(original)

    replay_traj = Trajectory(agent_name="a", scenario_name="s")
    call0 = schedule.maybe_inject({"ok": True}, replay_traj)
    call1 = schedule.maybe_inject({"ok": True}, replay_traj)
    call2 = schedule.maybe_inject({"ok": True}, replay_traj)

    assert call0 == {"ok": True}  # untouched
    assert call1 == {"error": "boom"}  # fault correctly lands on execution 1
    assert call2 == {"ok": True}  # untouched


def test_multiple_faults_on_one_call_are_batched_and_replayed_together():
    original = _trajectory_with_raw_steps(
        [
            (StepType.TOOL_CALL, {"tool": "a"}),
            (StepType.FAULT_INJECTED, {"fault": "malformed", "mutated": {"corrupted": True}}),
            (StepType.FAULT_INJECTED, {"fault": "tool_error", "mutated": {"error": "boom"}}),
            (StepType.TOOL_RESULT, {"error": "boom"}),
        ]
    )

    schedule = DeterministicFaultSchedule.from_trajectory(original)

    assert schedule._batches == [
        FaultBatch(
            faults=[
                {"fault": "malformed", "mutated": {"corrupted": True}},
                {"fault": "tool_error", "mutated": {"error": "boom"}},
            ]
        )
    ]

    replay_traj = Trajectory(agent_name="a", scenario_name="s")
    result = schedule.maybe_inject({"ok": True}, replay_traj)

    # both faults applied in order — the second's "mutated" wins, since
    # each fault in the batch is applied in sequence to the running value
    assert result == {"error": "boom"}
    fault_steps = [s for s in replay_traj.steps if s.type == StepType.FAULT_INJECTED]
    assert len(fault_steps) == 2


def test_call_that_raised_before_chaos_gets_no_batch():
    """A TOOL_CALL followed directly by an ERROR (no TOOL_RESULT) means the
    wrapped tool raised before maybe_inject was ever reached — that
    execution must not consume a fault-batch slot on replay.
    """
    original = _trajectory_with_raw_steps(
        [
            (StepType.TOOL_CALL, {"tool": "a"}),
            (StepType.ERROR, "a raised: boom"),
            (StepType.TOOL_CALL, {"tool": "b"}),
            (StepType.FAULT_INJECTED, {"fault": "tool_error", "mutated": {"error": "x"}}),
            (StepType.TOOL_RESULT, {"error": "x"}),
        ]
    )

    schedule = DeterministicFaultSchedule.from_trajectory(original)

    # Only ONE batch — for execution "b" — since "a" never reached maybe_inject.
    assert len(schedule._batches) == 1
    replay_traj = Trajectory(agent_name="a", scenario_name="s")
    assert schedule.maybe_inject({"ok": True}, replay_traj) == {"error": "x"}


def test_call_beyond_recorded_batches_passes_through_unmodified():
    schedule = DeterministicFaultSchedule.from_trajectory(_trajectory_with_raw_steps([]))
    replay_traj = Trajectory(agent_name="a", scenario_name="s")

    assert schedule.maybe_inject({"ok": True}, replay_traj) == {"ok": True}


# ---------------------------------------------------------------------------
# Flagship integration test: broken agent -> reproduced -> patched -> fixed
# ---------------------------------------------------------------------------


def _order_status_judge() -> TrajectoryJudge:
    return TrajectoryJudge(
        expected_tool_calls=[{"tool": "lookup_order", "args": {"order_id": "98765"}}]
    )


def _make_chaos() -> ChaosEngine:
    # Two faults, probability 1.0, so every tool execution deterministically
    # gets a two-fault batch — exercises the multi-fault-per-call path for
    # real, not just in the synthetic unit tests above.
    return ChaosEngine(
        faults=[MalformedResponseFault(probability=1.0), ToolErrorFault(probability=1.0)],
        seed=1,
    )


def test_broken_agent_fails_and_replay_faithfully_reproduces_it():
    broken_agent = CustomerSupportAgent(failure_mode="wrong_order_id")
    harness = Harness(
        agent=broken_agent,
        tools=TOOLS,
        chaos_engine=_make_chaos(),
        evaluators=[_order_status_judge()],
    )
    original_run = harness.run(
        Scenario(name="order-status", initial_input="What's the status of order #98765?")
    )

    assert original_run.verdict == "FAIL"  # wrong_order_id calls lookup_order(order_id="99999")

    # Sanity check the chaos engine actually did fire two faults per call —
    # otherwise this test wouldn't be exercising FaultBatch at all.
    tool_result_count = len(
        [s for s in original_run.trajectory.steps if s.type == StepType.TOOL_RESULT]
    )
    fault_count = len(
        [s for s in original_run.trajectory.steps if s.type == StepType.FAULT_INJECTED]
    )
    assert fault_count == 2 * tool_result_count
    assert tool_result_count > 0

    session = ReplaySession(
        agent=CustomerSupportAgent(failure_mode="wrong_order_id"),
        tools=TOOLS,
        evaluators=[_order_status_judge()],
    )
    replay_result = session.replay(original_run)

    assert replay_result.reproduced_failure
    assert not replay_result.fixed
    # Replaying the SAME agent with the SAME recorded fault schedule should
    # be byte-for-byte identical — this is the payoff of fixing FaultBatch
    # alignment: zero divergences on an unchanged replay.
    assert replay_result.divergences == []


def test_patched_agent_resolves_the_failure_on_replay():
    """Same recorded failing run as above, but replayed against the
    DEFAULT (non-broken) agent — the fix-verification workflow.
    """
    broken_agent = CustomerSupportAgent(failure_mode="wrong_order_id")
    harness = Harness(
        agent=broken_agent,
        tools=TOOLS,
        chaos_engine=_make_chaos(),
        evaluators=[_order_status_judge()],
    )
    original_run = harness.run(
        Scenario(name="order-status", initial_input="What's the status of order #98765?")
    )
    assert original_run.verdict == "FAIL"

    patched_session = ReplaySession(
        agent=CustomerSupportAgent(),  # no failure_mode — the "fix"
        tools=TOOLS,
        evaluators=[_order_status_judge()],
    )
    replay_result = patched_session.replay(original_run)

    assert replay_result.fixed
    assert not replay_result.reproduced_failure
    # The patched agent calls a DIFFERENT tool argument than the broken one
    # did, so divergence from the original recording is expected here —
    # unlike the unchanged-replay case above.
    assert replay_result.divergences != []
