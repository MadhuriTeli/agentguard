from agentguard.contracts.schema import Contract
from agentguard.core.result import FailureType, Verdict
from agentguard.core.trajectory import StepType, Trajectory
from agentguard.judges.rule_judge import RuleJudge


def _trajectory_with_message(text: str) -> Trajectory:
    traj = Trajectory(agent_name="a", scenario_name="s")
    traj.add_step(StepType.MESSAGE, text, role="assistant")
    return traj


def test_passes_when_no_forbidden_pattern_matches():
    judge = RuleJudge(forbidden_patterns=[r"\bguarantee\b"])
    result = judge.evaluate(_trajectory_with_message("Your order will ship soon."))

    assert result.verdict == Verdict.PASS
    assert result.evidence == []


def test_fails_with_evidence_when_pattern_matches():
    judge = RuleJudge(forbidden_patterns=[r"\bguarantee\b"])
    result = judge.evaluate(_trajectory_with_message("I guarantee this will work."))

    assert result.verdict == Verdict.FAIL
    assert result.failure_types == [FailureType.POLICY_VIOLATION]
    assert "guarantee" in result.evidence[0].message


def test_contract_forbidden_phrases_are_included():
    judge = RuleJudge()
    contract = Contract(name="c", forbidden_phrases=[r"\brefund\b"])
    result = judge.evaluate(_trajectory_with_message("Sure, refund on the way."), contract)

    assert result.verdict == Verdict.FAIL
    assert len(result.evidence) == 1


def test_multiple_matched_patterns_produce_multiple_evidence_entries():
    judge = RuleJudge(forbidden_patterns=[r"\bguarantee\b", r"\bpromise\b"])
    result = judge.evaluate(_trajectory_with_message("I guarantee and promise this."))

    assert len(result.evidence) == 2
    assert result.failure_types == [FailureType.POLICY_VIOLATION]
