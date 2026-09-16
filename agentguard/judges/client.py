"""A production-grade LLMClient backed by the Anthropic SDK, for LLMJudge.

Usage:
    from agentguard.judges.client import Client
    from agentguard.judges.llm_judge import LLMJudge

    client = Client(model="claude-sonnet-4-6")
    judge = LLMJudge(llm_client=client, rubric="The agent must not leak PII.")

Requires the ANTHROPIC_API_KEY environment variable to be set (the SDK
reads it automatically), or pass api_key explicitly.

Why this isn't a five-line wrapper around `client.messages.create`:

- Grading calls run inside a CI gate. A single transient 429/500 from the
  API shouldn't fail your whole build — so this retries transient errors
  with exponential backoff before giving up.
- The judge's grading instructions live in a `system` prompt, not crammed
  into the user turn. This is a meaningfully more reliable way to get an
  instruction-following model (like "respond with ONLY JSON") to actually
  comply, and keeps LLMJudge's per-call prompt focused on the task content.
- Errors are normalized to LLMClientError so LLMJudge (and anything else
  that depends on the LLMClient protocol) never needs to know this is
  Anthropic specifically, or handle SDK-specific exception types.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic

from agentguard.judges.llm_judge import JUDGE_SYSTEM_PROMPT

logger = logging.getLogger("agentguard.judges.client")


class LLMClientError(RuntimeError):
    """Raised when the underlying LLM call fails after exhausting retries."""


@dataclass
class Client:
    """Thin, resilient adapter from the Anthropic SDK to LLMJudge's `complete`."""

    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    api_key: str | None = None
    system_prompt: str = JUDGE_SYSTEM_PROMPT

    _client: anthropic.Anthropic = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Imported lazily so the rest of AgentGuard doesn't require the
        # anthropic package unless you actually use this client.
        import anthropic

        # client_kwargs: dict[str, object] = {"timeout": self.timeout_seconds}
        if self.api_key is not None:
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.timeout_seconds,
            )
        else:
            self._client = anthropic.Anthropic(
                timeout=self.timeout_seconds,
            )

        # self._client = anthropic.Anthropic(**client_kwargs)
        self._anthropic = anthropic

    def complete(self, prompt: str) -> str:
        """Send `prompt` to the model and return its text response.

        Retries transient failures (rate limits, connection errors, 5xxs)
        with exponential backoff. Raises LLMClientError if every attempt
        fails, or if the response contains no text content.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                return self._extract_text(response)

            except self._anthropic.APIStatusError as exc:
                last_error = exc
                if exc.status_code not in (429, 500, 502, 503, 529):
                    # Non-transient (e.g. 400 bad request, 401 auth) — retrying
                    # won't help and would just waste time/quota.
                    raise LLMClientError(
                        f"Anthropic API returned a non-retryable error: {exc}"
                    ) from exc
                logger.warning(
                    "Anthropic API call failed (attempt %d/%d, status=%s): %s",
                    attempt,
                    self.max_retries,
                    exc.status_code,
                    exc,
                )

            except self._anthropic.APIConnectionError as exc:
                last_error = exc
                logger.warning(
                    "Anthropic API connection error (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                delay = self.retry_base_delay_seconds * (2 ** (attempt - 1))
                time.sleep(delay)

        raise LLMClientError(
            f"Anthropic API call failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _extract_text(response: object) -> str:
        text_blocks = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        if not text_blocks:
            raise LLMClientError(f"Model response contained no text content: {response!r}")
        return "".join(text_blocks)
