from agentguard.core.result import EvalResult, RunResult, Verdict
from agentguard.regression.comparator import RegressionComparator


def make_run(scenario: str, eval_name: str, verdict: Verdict) -> RunResult:
    return RunResult(
        trajectory_id="t1",
        agent_name="agent",
        scenario_name=scenario,
        eval_results=[EvalResult(name=eval_name, verdict=verdict)],
    )


def test_detects_regression():
    baseline = [make_run("scenario_a", "safety", Verdict.PASS)]
    candidate = [make_run("scenario_a", "safety", Verdict.FAIL)]

    report = RegressionComparator().compare(baseline, candidate)

    assert report.has_regressions
    assert report.regressions[0].scenario_name == "scenario_a"


def test_detects_improvement_not_regression():
    baseline = [make_run("scenario_a", "safety", Verdict.FAIL)]
    candidate = [make_run("scenario_a", "safety", Verdict.PASS)]

    report = RegressionComparator().compare(baseline, candidate)

    assert not report.has_regressions
    assert len(report.improvements) == 1
