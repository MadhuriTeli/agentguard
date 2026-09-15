from agentguard.core.replay import ReplaySession
from agentguard.core.runner import Runner, Scenario
from agentguard.core.trajectory import StepType, Trajectory


class _EchoAgent:
    """Agent that emits one 'reply' tool call per step, no faults."""

    name = "echo_agent"
    calls = 0

    def step(self, observation):
        self.calls += 1
        return {"tool": "reply", "args": {"message": f"call-{self.calls}"}}


class _RecordingRunner(Runner):
    """Runner whose _execute just echoes the action back as the observation,
    and marks the trajectory done after one tool call — enough to exercise
    replay without needing a real tool-execution layer.
    """

    def run(self, scenario: Scenario) -> Trajectory:
        trajectory = Trajectory(agent_name=self.agent.name, scenario_name=scenario.name)
        trajectory.add_step(StepType.MESSAGE, scenario.initial_input, role="user")

        action = self.agent.step(scenario.initial_input)
        trajectory.add_step(StepType.TOOL_CALL, action)
        trajectory.add_step(StepType.TOOL_RESULT, {"ok": True})
        trajectory.finish(success=False)  # pretend this scenario failed
        return trajectory


def _make_original_trajectory() -> Trajectory:
    agent = _EchoAgent()
    runner = _RecordingRunner(agent=agent)
    return runner.run(Scenario(name="s1", initial_input="hello"))


def test_replay_reproduces_same_shape_trajectory(monkeypatch):
    original = _make_original_trajectory()

    # Patch ReplaySession to use our lightweight runner instead of the real
    # tool-execution-dependent Runner.
    session = ReplaySession(agent=_EchoAgent())
    monkeypatch.setattr(
        "agentguard.core.replay.Runner", _RecordingRunner
    )

    result = session.replay(original)

    assert result.replayed_trajectory.agent_name == "echo_agent"
    assert result.replayed_trajectory.tool_calls()
    assert isinstance(result.divergences, list)


def test_fixed_property_true_when_replay_passes():
    original = _make_original_trajectory()
    replayed = Trajectory(agent_name="echo_agent", scenario_name="s1")
    replayed.finish(success=True)

    from agentguard.core.replay import ReplayResult

    result = ReplayResult(
        original_trajectory=original, replayed_trajectory=replayed, divergences=[]
    )

    assert result.fixed
    assert not result.reproduced_failure
