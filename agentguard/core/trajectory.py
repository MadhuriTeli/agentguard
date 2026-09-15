"""Data structures for capturing an agent's run as a trajectory.

A trajectory is the full, ordered record of what an agent did during a
single scenario run: the messages it exchanged, the tools it called, the
observations it received, and any errors or faults encountered along the way.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StepType(str, Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    FAULT_INJECTED = "fault_injected"


@dataclass
class Step:
    """A single event within a trajectory."""

    type: StepType
    content: Any
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """The complete recorded run of an agent against one scenario."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = ""
    scenario_name: str = ""
    steps: list[Step] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    success: Optional[bool] = None

    def add_step(self, step_type: StepType, content: Any, **metadata: Any) -> Step:
        step = Step(type=step_type, content=content, metadata=metadata)
        self.steps.append(step)
        return step

    def finish(self, success: bool) -> None:
        self.ended_at = time.time()
        self.success = success

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.type == StepType.TOOL_CALL]
