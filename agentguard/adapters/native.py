"""Adapter for AgentGuard's own pre-existing agent interface.

Wraps any object exposing `step(observation) -> {"tool": ..., "args": {}}`
(the shape every agent in this codebase used before AgentAdapter existed,
including CustomerSupportAgent) so it satisfies the AgentAdapter protocol
with zero changes required on the agent's side.
"""

from __future__ import annotations

from typing import Any

from agentguard.adapters.base import AgentExecution, ExecutionContext


class NativeAgentAdapter:
    """Wraps a step()-style agent to satisfy AgentAdapter."""

    def __init__(self, agent: Any):
        self._agent = agent

    @property
    def name(self) -> str:
        return getattr(self._agent, "name", self._agent.__class__.__name__)

    def run(self, observation: Any, context: ExecutionContext) -> AgentExecution:
        action = self._agent.step(observation)
        return AgentExecution(action=action)
