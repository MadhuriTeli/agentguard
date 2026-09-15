"""Concrete fault types that the ChaosEngine can inject.

Each Fault has a probability of firing and an `apply` method that mutates
the observation/tool-result being passed to the agent, simulating a
real-world failure mode.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class Fault(ABC):
    name: str = "fault"
    probability: float = 0.0

    @abstractmethod
    def apply(self, observation: Any) -> Any:
        raise NotImplementedError


@dataclass
class LatencyFault(Fault):
    """Simulates a slow tool/service response."""

    name: str = "latency"
    probability: float = 0.1
    delay_seconds: float = 2.0

    def apply(self, observation: Any) -> Any:
        time.sleep(self.delay_seconds)
        return observation


@dataclass
class ToolErrorFault(Fault):
    """Simulates a tool call failing outright."""

    name: str = "tool_error"
    probability: float = 0.1
    error_message: str = "Simulated tool failure"

    def apply(self, observation: Any) -> Any:
        return {"error": self.error_message}


@dataclass
class MalformedResponseFault(Fault):
    """Simulates a tool returning malformed / unexpected data."""

    name: str = "malformed_response"
    probability: float = 0.1

    def apply(self, observation: Any) -> Any:
        if isinstance(observation, dict):
            return {**observation, "__corrupted__": True}
        return None


@dataclass
class EmptyResponseFault(Fault):
    """Simulates a tool returning an empty/null result."""

    name: str = "empty_response"
    probability: float = 0.05

    def apply(self, observation: Any) -> Any:
        return None
