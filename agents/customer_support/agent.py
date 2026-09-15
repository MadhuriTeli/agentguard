"""Deterministic customer-support agent used by AgentGuard evaluations."""

from __future__ import annotations

import re
from typing import Any


class CustomerSupportAgent:
    name = "customer_support_agent"

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
        self._last_tool: str | None = None
        self._finished = False

    def step(self, observation: Any) -> dict:
        """Choose the next tool call from the current observation."""

        # ---------------------------------------------------------
        # After a completed operation, produce the final response.
        # ---------------------------------------------------------
        if self._finished:
            return {
                "tool": "reply",
                "args": {
                    "message": self._build_final_response(observation),
                },
            }

        text = self._extract_text(observation).lower()

        # ---------------------------------------------------------
        # Continue a return-policy lookup
        # ---------------------------------------------------------
        if self._last_tool == "lookup_policy":
            self._finished = True

            return {
                "tool": "reply",
                "args": {
                    "message": self._build_final_response(observation),
                },
            }

        # ---------------------------------------------------------
        # Continue an order lookup
        # ---------------------------------------------------------
        if self._last_tool == "lookup_order":
            self._finished = True

            return {
                "tool": "reply",
                "args": {
                    "message": self._build_final_response(observation),
                },
            }

        # ---------------------------------------------------------
        # Continue after identity verification
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Continue after confirmation
        # ---------------------------------------------------------
        if self._last_tool == "confirm_with_user":
            self._last_tool = "cancel_subscription"

            return {
                "tool": "cancel_subscription",
                "args": {},
            }

        # ---------------------------------------------------------
        # Continue after cancellation
        # ---------------------------------------------------------
        if self._last_tool == "cancel_subscription":
            self._finished = True

            return {
                "tool": "reply",
                "args": {
                    "message": "Your subscription has been cancelled.",
                },
            }

        # ---------------------------------------------------------
        # PII request
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Return policy
        # ---------------------------------------------------------
        if "return policy" in text or "return" in text:
            self._last_tool = "lookup_policy"

            return {
                "tool": "lookup_policy",
                "args": {
                    "topic": "returns",
                },
            }

        # ---------------------------------------------------------
        # Order status
        # ---------------------------------------------------------
        if "order" in text and (
            "status" in text
            or "arrived" in text
            or "where is" in text
        ):
            match = re.search(r"#?(\d{5,})", text)
            order_id = match.group(1) if match else "unknown"

            self._last_tool = "lookup_order"

            return {
                "tool": "lookup_order",
                "args": {
                    "order_id": order_id,
                },
            }

        # ---------------------------------------------------------
        # Refund
        # ---------------------------------------------------------
        if "refund" in text:
            self._last_tool = "verify_identity"

            return {
                "tool": "verify_identity",
                "args": {},
            }

        # ---------------------------------------------------------
        # Subscription cancellation
        # ---------------------------------------------------------
        if "cancel" in text and "subscription" in text:
            self._last_tool = "confirm_with_user"

            return {
                "tool": "confirm_with_user",
                "args": {},
            }

        # ---------------------------------------------------------
        # Fallback
        # ---------------------------------------------------------
        self._finished = True

        return {
            "tool": "reply",
            "args": {
                "message": (
                    "I can help with orders, returns, refunds, "
                    "and subscription requests."
                ),
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
                    return (
                        f"Your order is {status}. "
                        f"The estimated delivery is {eta}."
                    )

                return f"Your order status is {status}."

        return "Your request has been processed."
