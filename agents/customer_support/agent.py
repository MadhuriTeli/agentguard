"""Deterministic customer-support agent used by AgentGuard evaluations."""

from __future__ import annotations

import re
from typing import Any


class CustomerSupportAgent:
    name = "customer_support_agent"

    def __init__(
        self,
        llm_client: Any = None,
        failure_mode: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.failure_mode = failure_mode
        self._last_tool: str | None = None
        self._finished: bool = False

    def _failure_response(self, text: str) -> dict[str, Any] | None:
        """Return an intentionally bad action when failure injection is enabled."""

        if self.failure_mode == "wrong_order_id" and "order" in text:
            self._last_tool = "lookup_order"
            return {
                "tool": "lookup_order",
                "args": {
                    "order_id": "99999",
                },
            }

        if self.failure_mode == "wrong_tool" and "order" in text:
            self._last_tool = "lookup_policy"
            return {
                "tool": "lookup_policy",
                "args": {
                    "topic": "returns",
                },
            }

        if self.failure_mode == "skip_confirmation" and "cancel" in text:
            self._last_tool = "cancel_subscription"
            return {
                "tool": "cancel_subscription",
                "args": {},
            }

        if self.failure_mode == "unsafe_pii" and (
            "another customer" in text
            or "account details" in text
            or "guess their email" in text
            or "private data" in text
        ):
            self._finished = True
            return {
                "tool": "lookup_customer_by_email",
                "args": {
                    "email": "victim@example.com",
                },
            }

        if self.failure_mode == "wrong_response":
            self._finished = True
            return {
                "tool": "reply",
                "args": {
                    "message": "I don't know the answer.",
                },
            }

        return None

    def step(self, observation: Any) -> dict[str, Any]:
        """Choose the next tool call based on the current observation."""

        if self._finished:
            return {
                "tool": "reply",
                "args": {
                    "message": self._build_final_response(observation),
                },
            }

        text = self._extract_text(observation).lower()

        # Intentionally inject a bad behavior when requested.
        failure = self._failure_response(text)

        if failure is not None:
            return failure

        if self._last_tool == "lookup_policy":
            self._finished = True
            return {
                "tool": "reply",
                "args": {
                    "message": self._build_final_response(observation),
                },
            }

        if self._last_tool == "lookup_order":
            self._finished = True
            return {
                "tool": "reply",
                "args": {
                    "message": self._build_final_response(observation),
                },
            }

        if self._last_tool == "verify_identity":
            self._finished = True
            return {
                "tool": "reply",
                "args": {
                    "message": (
                        "I can't issue a refund without completing the "
                        "required verification and refund checks."
                    ),
                },
            }

        if self._last_tool == "confirm_with_user":
            self._last_tool = "cancel_subscription"
            return {
                "tool": "cancel_subscription",
                "args": {},
            }

        if self._last_tool == "cancel_subscription":
            self._finished = True
            return {
                "tool": "reply",
                "args": {
                    "message": "Your subscription has been cancelled.",
                },
            }

        if (
            "another customer" in text
            or "account details" in text
            or "guess their email" in text
            or "private data" in text
        ):
            self._finished = True
            return {
                "tool": "reply",
                "args": {
                    "message": (
                        "I can't provide another customer's private "
                        "account details or personal information."
                    ),
                },
            }

        if "return policy" in text or "return" in text:
            self._last_tool = "lookup_policy"
            return {
                "tool": "lookup_policy",
                "args": {
                    "topic": "returns",
                },
            }

        if "order" in text and ("status" in text or "arrived" in text or "where is" in text):
            match = re.search(r"#?(\d{5,})", text)
            order_id = match.group(1) if match else "unknown"

            self._last_tool = "lookup_order"

            return {
                "tool": "lookup_order",
                "args": {
                    "order_id": order_id,
                },
            }

        if "refund" in text:
            self._last_tool = "verify_identity"
            return {
                "tool": "verify_identity",
                "args": {},
            }

        if "cancel" in text and "subscription" in text:
            self._last_tool = "confirm_with_user"
            return {
                "tool": "confirm_with_user",
                "args": {},
            }

        self._finished = True

        return {
            "tool": "reply",
            "args": {
                "message": ("I can help with orders, returns, refunds, and subscription requests."),
            },
        }

    @staticmethod
    def _extract_text(observation: Any) -> str:
        if isinstance(observation, str):
            return observation

        if isinstance(observation, dict):
            if "message" in observation:
                return str(observation["message"])

            return str(observation)

        return str(observation)

    @staticmethod
    def _build_final_response(observation: Any) -> str:
        if isinstance(observation, dict):
            if "return_window_days" in observation:
                return (
                    f"Our return policy allows returns within "
                    f"{observation['return_window_days']} days of purchase."
                )

            if "status" in observation:
                status = observation["status"]
                eta = observation.get("eta")

                if eta:
                    return f"Your order is {status}. The estimated delivery is {eta}."

                return f"Your order status is {status}."

        return "Your request has been processed."
