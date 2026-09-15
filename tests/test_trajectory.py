from agentguard.core.trajectory import StepType, Trajectory


def test_add_step_appends_and_returns_step():
    traj = Trajectory(agent_name="test_agent", scenario_name="test_scenario")
    step = traj.add_step(StepType.MESSAGE, "hello", role="user")

    assert step in traj.steps
    assert step.content == "hello"
    assert step.metadata["role"] == "user"


def test_finish_sets_success_and_duration():
    traj = Trajectory()
    traj.finish(success=True)

    assert traj.success is True
    assert traj.duration_seconds is not None
    assert traj.duration_seconds >= 0


def test_tool_calls_filters_correctly():
    traj = Trajectory()
    traj.add_step(StepType.MESSAGE, "hi")
    traj.add_step(StepType.TOOL_CALL, {"tool": "lookup_order"})
    traj.add_step(StepType.TOOL_RESULT, {"status": "shipped"})

    calls = traj.tool_calls()
    assert len(calls) == 1
    assert calls[0].content["tool"] == "lookup_order"
