"""Schema for agent behavior contracts.

A Contract declaratively describes what an agent is and isn't allowed to do
during a scenario: which tools it may call, steps it must take, and actions
that are strictly forbidden (e.g. issuing a refund without verification).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ToolConstraint(BaseModel):
    tool_name: str
    required: bool = False
    max_calls: int | None = None
    forbidden_args: dict[str, list[str]] = Field(default_factory=dict)


class Contract(BaseModel):
    """Declarative spec of expected/allowed agent behavior for a scenario."""

    name: str
    description: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    tool_constraints: list[ToolConstraint] = Field(default_factory=list)
    required_steps: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    max_turns: int | None = None

    model_config = ConfigDict(extra="forbid")
