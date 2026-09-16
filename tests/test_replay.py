from agentguard.core.replay import ReplaySession
from agentguard.core.runner import Runner, Scenario
from agentguard.core.trajectory import StepType, Trajectory


class _EchoAgent:
    """Agent that emits one reply tool call."""

    name = "echo_agent"
    calls = 0

    def step(self, observation):
        self.calls += 1
        return {"tool": "reply", "args": {"message": f"call-{self.calls}"}}


class _RecordingRunner(Runner):
    """Lightweight runner used to create a deterministic original trajectory."""

    def run(self, scenario: Scenario) -> Trajectory:
        trajectory = Trajectory(
            agent_name=self.agent.name,
            scenario_name=scenario.name,
            initial_input=scenario.initial_input,
        )

        trajectory.add_step(
            StepType.MESSAGE,
            scenario.initial_input,
            role="user",
        )

        action = self.agent.step(scenario.initial_input)
        trajectory.add_step(StepType.TOOL_CALL, action)
        trajectory.add_step(StepType.TOOL_RESULT, {"ok": True})
        trajectory.finish(success=False)

        return trajectory


def _make_original_trajectory() -> Trajectory:
    agent = _EchoAgent()
    runner = _RecordingRunner(agent=agent)

    return runner.run(
        Scenario(
            name="s1",
            initial_input="hello",
        )
    )


def test_replay_session_requires_real_harness_dependencies():
    session = ReplaySession(
        agent=_EchoAgent(),
        tools={
            "reply": lambda message: {"message": message},
        },
    )

    assert session.agent.name == "echo_agent"
    assert "reply" in session.tools


def test_fixed_property_true_when_replay_passes():
    original = _make_original_trajectory()

    replayed = Trajectory(
        agent_name="echo_agent",
        scenario_name="s1",
        initial_input="hello",
    )
    replayed.finish(success=True)

    from agentguard.core.replay import ReplayResult

    result = ReplayResult(
        original_trajectory=original,
        replayed_trajectory=replayed,
        divergences=[],
    )

    assert result.fixed
    assert not result.reproduced_failure
