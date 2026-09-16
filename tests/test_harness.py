from agentguard.chaos.engine import ChaosEngine
from agentguard.chaos.faults import ToolErrorFault
from agentguard.contracts.schema import Contract, ToolConstraint
from agentguard.core.harness import Harness
from agentguard.core.result import EvalResult, RunResult, Verdict
from agentguard.core.runner import Scenario


class _ScriptedAgent:
    """Agent that plays back a fixed list of actions, one per step() call."""

    name = "scripted_agent"

    def __init__(self, actions: list[dict]):
        self._actions = list(actions)

    def step(self, observation):
        return self._actions.pop(0)


def _done_after_n_tool_calls(n: int):
    def is_done(trajectory):
        return len(trajectory.tool_calls()) >= n

    return is_done


def test_happy_path_passes_with_no_contract():
    tools = {"reply": lambda message: {"message": message}}
    agent = _ScriptedAgent([{"tool": "reply", "args": {"message": "hi"}}])
    harness = Harness(agent=agent, tools=tools)

    result = harness.run(
        Scenario(name="s1", initial_input="hello", is_done=_done_after_n_tool_calls(1))
    )

    assert result.passed
    assert result.verdict == "PASS"
    assert result.metrics.num_tool_calls == 1


def test_forbidden_tool_is_blocked_live_and_fails():
    tools = {
        "reply": lambda message: {"message": message},
        "issue_refund": lambda amount: {"refunded": amount},
    }
    agent = _ScriptedAgent([{"tool": "issue_refund", "args": {"amount": 10000}}])
    contract = Contract(name="no_refunds", forbidden_tools=["issue_refund"])
    harness = Harness(agent=agent, tools=tools, contract=contract)

    result = harness.run(Scenario(name="s2", initial_input="refund me"))

    assert not result.passed
    assert result.verdict == "FAIL"
    assert result.contract_result is not None
    assert result.contract_result.verdict == Verdict.FAIL
    assert "issue_refund" in result.contract_result.reason
    # the refund tool must never have actually executed
    assert result.metrics.tool_call_counts.get("issue_refund") == 1
    assert result.metrics.num_live_violations == 1


def test_max_calls_enforced_live():
    call_log = []
    tools = {"lookup_order": lambda order_id: call_log.append(order_id) or {"status": "ok"}}
    agent = _ScriptedAgent(
        [
            {"tool": "lookup_order", "args": {"order_id": "1"}},
            {"tool": "lookup_order", "args": {"order_id": "1"}},
        ]
    )
    contract = Contract(
        name="single_lookup",
        tool_constraints=[ToolConstraint(tool_name="lookup_order", max_calls=1)],
    )
    harness = Harness(agent=agent, tools=tools, contract=contract)

    result = harness.run(Scenario(name="s3", initial_input="where is my order"))

    assert not result.passed
    # second call should never have reached the real tool fn
    assert call_log == ["1"]


def test_chaos_engine_injects_fault_into_tool_result():
    tools = {"flaky_tool": lambda: {"status": "ok"}}
    agent = _ScriptedAgent([{"tool": "flaky_tool", "args": {}}])
    chaos = ChaosEngine(faults=[ToolErrorFault(probability=1.0)])
    harness = Harness(agent=agent, tools=tools, chaos_engine=chaos)

    result = harness.run(
        Scenario(name="s4", initial_input="go", is_done=_done_after_n_tool_calls(1))
    )

    assert result.metrics.num_faults_injected == 1


def test_regression_against_baseline_fails_overall_verdict():
    tools = {"reply": lambda message: {"message": message}}
    agent = _ScriptedAgent([{"tool": "reply", "args": {"message": "hi"}}])

    baseline = RunResult(
        trajectory_id="baseline-1",
        agent_name="scripted_agent",
        scenario_name="s5",
        eval_results=[EvalResult(name="safety_judge", verdict=Verdict.PASS)],
    )

    class _AlwaysFailJudge:
        name = "safety_judge"

        def evaluate(self, trajectory, contract=None):
            return EvalResult(name="safety_judge", verdict=Verdict.FAIL, reason="regressed")

    harness = Harness(
        agent=agent,
        tools=tools,
        evaluators=[_AlwaysFailJudge()],
        baseline=baseline,
    )

    result = harness.run(
        Scenario(name="s5", initial_input="hello", is_done=_done_after_n_tool_calls(1))
    )

    assert not result.passed
    assert any(e.is_regression for e in result.regression_entries)


def test_reply_tool_call_is_recorded_as_assistant_message():
    """TrajectoryJudge's final_response_contains check depends entirely on
    this: a 'reply' tool call must show up as a MESSAGE step tagged
    role=assistant, not just as a TOOL_CALL/TOOL_RESULT pair. If this
    regresses, TrajectoryJudge silently stops being able to check final
    responses at all (see test_trajectory_judge.py).
    """
    from agentguard.core.trajectory import StepType

    tools = {"reply": lambda message: {"message": message}}
    agent = _ScriptedAgent([{"tool": "reply", "args": {"message": "Your order has shipped."}}])
    harness = Harness(agent=agent, tools=tools)

    result = harness.run(
        Scenario(name="s6", initial_input="hello", is_done=_done_after_n_tool_calls(1))
    )

    assistant_messages = [
        step
        for step in result.trajectory.steps
        if step.type == StepType.MESSAGE and step.metadata.get("role") == "assistant"
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == "Your order has shipped."
