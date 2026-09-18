from agentguard.contracts.schema import Contract
from agentguard.core.result import FailureType, Verdict
from agentguard.core.trajectory import StepType, Trajectory
from agentguard.judges.llm_judge import LLMJudge


class _StubClient:
    """A controllable stand-in for a real LLMClient."""

    def __init__(self, response: str | None = None, raises: Exception | None = None):
        self.response = response
        self.raises = raises
        self.last_prompt: str | None = None
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        if self.raises is not None:
            raise self.raises
        return self.response if self.response is not None else ""


def _make_trajectory() -> Trajectory:
    traj = Trajectory(agent_name="a", scenario_name="s")
    traj.add_step(StepType.MESSAGE, "hello")
    return traj


def test_pass_verdict_parsed_correctly():
    client = _StubClient('{"verdict": "pass", "reason": "Looks good"}')
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.PASS
    assert result.reason == "Looks good"


def test_fail_verdict_parsed_correctly():
    client = _StubClient('{"verdict": "fail", "reason": "Leaked PII"}')
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.FAIL
    assert result.reason == "Leaked PII"


def test_verdict_value_is_case_and_whitespace_insensitive():
    client = _StubClient('{"verdict": "  PASS  ", "reason": "ok"}')
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.PASS


def test_markdown_json_fence_is_stripped():
    client = _StubClient('```json\n{"verdict": "pass", "reason": "fenced"}\n```')
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.PASS
    assert result.reason == "fenced"


def test_bare_markdown_fence_without_json_tag_is_stripped():
    client = _StubClient('```\n{"verdict": "fail", "reason": "still fenced"}\n```')
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.FAIL


def test_non_json_response_yields_error_not_fail():
    """A formatting failure must not silently count as the agent failing."""
    client = _StubClient("I think this passes, no JSON here")
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.ERROR
    assert "not valid json" in result.reason.lower()


def test_json_without_verdict_key_yields_error_not_fail():
    client = _StubClient('{"reason": "forgot the verdict key"}')
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.ERROR
    assert "verdict" in result.reason.lower()


def test_unrecognized_verdict_value_yields_error_not_fail():
    client = _StubClient('{"verdict": "maybe", "reason": "unsure"}')
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.ERROR


def test_json_array_instead_of_object_yields_error():
    client = _StubClient('["pass", "looks fine"]')
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.ERROR


def test_client_exception_yields_error_not_crash():
    client = _StubClient(raises=RuntimeError("network timeout"))
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.ERROR
    assert "network timeout" in result.reason


def test_rubric_and_contract_description_are_included_in_prompt():
    client = _StubClient('{"verdict": "pass", "reason": "ok"}')
    contract = Contract(name="c", description="Never mention competitors.")

    LLMJudge(llm_client=client, rubric="Be nice.").evaluate(_make_trajectory(), contract=contract)

    assert "Be nice." in client.last_prompt
    assert "Never mention competitors." in client.last_prompt


def test_missing_reason_defaults_to_empty_string():
    client = _StubClient('{"verdict": "pass"}')
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.PASS
    assert result.reason == ""


def test_fail_verdict_carries_evidence():
    client = _StubClient('{"verdict": "fail", "reason": "Leaked a customer email"}')
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert len(result.evidence) == 1
    assert result.evidence[0].message == "Leaked a customer email"
    assert result.failure_types == [FailureType.WRONG_FINAL_ANSWER]


def test_pass_verdict_carries_no_evidence():
    client = _StubClient('{"verdict": "pass", "reason": "Looks good"}')
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.evidence == []


def test_error_verdict_carries_no_evidence():
    """A malformed response is a judge malfunction, not a classified
    behavior failure — it shouldn't show up in a failure-type report."""
    client = _StubClient("not json at all")
    result = LLMJudge(llm_client=client).evaluate(_make_trajectory())

    assert result.verdict == Verdict.ERROR
    assert result.evidence == []
