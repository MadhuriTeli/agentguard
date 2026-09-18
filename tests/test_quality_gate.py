from agentguard.regression.gate import QualityGate, QualityGateConfig

from agentguard.contracts.schema import Contract
from agentguard.core.harness import Harness
from agentguard.core.runner import Scenario
from agentguard.judges.trajectory_judge import TrajectoryJudge


class _ScriptedAgent:
    def __init__(self, actions: list[dict]):
        self._actions = list(actions)

    def step(self, observation):
        return self._actions.pop(0) if self._actions else {"tool": "reply", "args": {}}


def _run(actions: list[dict], tools: dict, contract=None):
    num_actions = len(actions)
    harness = Harness(agent=_ScriptedAgent(actions), tools=tools, contract=contract)
    scenario = Scenario(
        name="s",
        initial_input="hi",
        is_done=lambda trajectory: len(trajectory.tool_calls()) >= num_actions,
    )
    return harness.run(scenario)


def test_gate_passes_with_default_config_when_everything_passes():
    tools = {"reply": lambda **kw: kw}
    runs = [_run([{"tool": "reply", "args": {}}], tools)]

    result = QualityGate().evaluate(runs)

    assert result.passed
    assert result.reasons == []


def test_gate_fails_on_pass_rate_below_minimum():
    tools = {"reply": lambda **kw: kw, "issue_refund": lambda **kw: kw}
    contract = Contract(name="c", forbidden_tools=["issue_refund"])
    runs = [
        _run([{"tool": "reply", "args": {}}], tools),
        _run([{"tool": "issue_refund", "args": {"amount": 5}}], tools, contract=contract),
    ]

    result = QualityGate().evaluate(runs)  # default min_pass_rate=1.0

    assert not result.passed
    assert any("pass rate" in r for r in result.reasons)


def test_gate_allows_configured_pass_rate_tolerance():
    """Use a TrajectoryJudge mismatch (AGENT_FAILURE category) rather than
    a contract violation, so this test isolates pass_rate tolerance from
    the separate max_contract_failures check — a contract-violating run
    would still trip that check even after loosening min_pass_rate.
    """
    tools = {"reply": lambda **kw: kw}
    judge = TrajectoryJudge(expected_tool_calls=[{"tool": "reply", "args": {"message": "x"}}])

    def _judged_run(message: str):
        harness = Harness(
            agent=_ScriptedAgent([{"tool": "reply", "args": {"message": message}}]),
            tools=tools,
            evaluators=[judge],
        )
        scenario = Scenario(
            name="s", initial_input="hi", is_done=lambda t: len(t.tool_calls()) >= 1
        )
        return harness.run(scenario)

    runs = [
        _judged_run("x"),
        _judged_run("x"),
        _judged_run("x"),
        _judged_run("wrong message"),  # fails TrajectoryJudge's arg match
    ]

    strict = QualityGate().evaluate(runs)
    assert not strict.passed

    lenient = QualityGate().evaluate(runs, config=QualityGateConfig(min_pass_rate=0.7))
    assert lenient.passed  # 3/4 = 75% clears a 70% bar


def test_gate_fails_on_contract_failures_exceeding_max():
    tools = {"reply": lambda **kw: kw, "issue_refund": lambda **kw: kw}
    contract = Contract(name="c", forbidden_tools=["issue_refund"])
    runs = [_run([{"tool": "issue_refund", "args": {"amount": 5}}], tools, contract=contract)]

    result = QualityGate().evaluate(
        runs, config=QualityGateConfig(min_pass_rate=0.0, max_contract_failures=0)
    )

    assert not result.passed
    assert any("contract failure" in r for r in result.reasons)


def test_gate_fails_on_regressions_exceeding_max():
    tools = {"reply": lambda **kw: kw}
    runs = [_run([{"tool": "reply", "args": {}}], tools)]

    result = QualityGate().evaluate(runs, num_regressions=2, config=QualityGateConfig())

    assert not result.passed
    assert any("regression" in r for r in result.reasons)


def test_gate_result_summary_is_json_serializable_shape():
    tools = {"reply": lambda **kw: kw}
    runs = [_run([{"tool": "reply", "args": {}}], tools)]

    result = QualityGate().evaluate(runs)
    summary = result.summary()

    assert summary["passed"] is True
    assert "metrics" in summary
    assert summary["num_regressions"] == 0
