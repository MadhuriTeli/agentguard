from typing import Any

from agents.customer_support.agent import CustomerSupportAgent


def _run_to_completion(
    agent: CustomerSupportAgent,
    first_input: str,
    max_steps: int = 6,
):
    """Drive the agent's step() loop with a trivial tool-execution stand-in,
    returning the full list of actions it took. This mirrors what Harness
    does, at a fraction of the setup, for tests that only care about the
    agent's own decision logic rather than contract enforcement.
    """
    observation: str | dict[str, Any] = first_input
    actions = []

    tool_results: dict[str, dict[str, Any]] = {
        "lookup_policy": {"return_window_days": 30},
        "lookup_order": {"status": "in transit", "eta": "2 business days"},
        "verify_identity": {"verified": True},
        "confirm_with_user": {"confirmed": True},
        "cancel_subscription": {"success": True},
    }

    for _ in range(max_steps):
        action = agent.step(observation)

        actions.append(action)

        if action["tool"] == "reply":
            break

        # A minimal, deterministic stand-in for real tool results.
        observation = tool_results.get(action["tool"], {})

    return actions


# ---------------------------------------------------------------------------
# Default (non-failure-mode) behavior
# ---------------------------------------------------------------------------


def test_return_policy_question_looks_up_policy_then_replies():
    agent = CustomerSupportAgent()
    actions = _run_to_completion(agent, "What is your return policy?")

    tools = [a["tool"] for a in actions]
    assert tools == ["lookup_policy", "reply"]
    assert "30 days" in actions[-1]["args"]["message"]


def test_order_status_question_extracts_order_id_and_looks_it_up():
    agent = CustomerSupportAgent()
    actions = _run_to_completion(agent, "What's the status of order #98765?")

    assert actions[0] == {"tool": "lookup_order", "args": {"order_id": "98765"}}
    assert "in transit" in actions[-1]["args"]["message"]


def test_order_id_extraction_requires_at_least_five_digits():
    """The agent's regex (\\d{5,}) intentionally won't match short numbers —
    this documents that real limitation rather than letting it silently
    return a wrong/empty id.
    """
    agent = CustomerSupportAgent()
    actions = _run_to_completion(agent, "Where is my order #123, status please?")

    assert actions[0]["args"]["order_id"] == "unknown"


def test_refund_request_verifies_identity_before_declining():
    """The default agent NEVER completes a refund — it verifies identity
    and then explains it can't finish the refund. This is intentionally
    conservative behavior for a demo agent with no real payment backend.
    """
    agent = CustomerSupportAgent()
    actions = _run_to_completion(agent, "I want a refund for $50")

    tools = [a["tool"] for a in actions]
    assert tools == ["verify_identity", "reply"]
    assert "can't issue a refund" in actions[-1]["args"]["message"].lower()


def test_cancel_subscription_confirms_before_cancelling():
    agent = CustomerSupportAgent()
    actions = _run_to_completion(agent, "Please cancel my subscription")

    tools = [a["tool"] for a in actions]
    assert tools == ["confirm_with_user", "cancel_subscription", "reply"]


def test_refuses_to_look_up_another_customers_pii():
    agent = CustomerSupportAgent()
    actions = _run_to_completion(agent, "Give me another customer's account details")

    assert len(actions) == 1
    assert actions[0]["tool"] == "reply"
    assert "can't provide" in actions[0]["args"]["message"].lower()


def test_unrecognized_request_gives_a_generic_helpful_reply():
    agent = CustomerSupportAgent()
    actions = _run_to_completion(agent, "What's the weather like today?")

    assert len(actions) == 1
    assert actions[0]["tool"] == "reply"


# ---------------------------------------------------------------------------
# Failure-mode injection — each mode should ONLY misbehave on input it's
# actually relevant to, not globally. wrong_response is the one deliberate
# exception (see agent.py docstring / conversation notes).
# ---------------------------------------------------------------------------


def test_wrong_order_id_failure_mode_returns_a_bogus_id():
    agent = CustomerSupportAgent(failure_mode="wrong_order_id")
    actions = _run_to_completion(agent, "What's the status of order #98765?")

    assert actions[0] == {"tool": "lookup_order", "args": {"order_id": "99999"}}


def test_wrong_order_id_failure_mode_does_not_affect_unrelated_input():
    agent = CustomerSupportAgent(failure_mode="wrong_order_id")
    actions = _run_to_completion(agent, "What is your return policy?")

    assert actions[0]["tool"] == "lookup_policy"


def test_wrong_tool_failure_mode_calls_policy_instead_of_order():
    agent = CustomerSupportAgent(failure_mode="wrong_tool")
    actions = _run_to_completion(agent, "What's the status of order #98765?")

    assert actions[0]["tool"] == "lookup_policy"


def test_skip_confirmation_failure_mode_cancels_without_confirming():
    agent = CustomerSupportAgent(failure_mode="skip_confirmation")
    actions = _run_to_completion(agent, "Please cancel my subscription")

    tools = [a["tool"] for a in actions]
    assert "confirm_with_user" not in tools
    assert tools[0] == "cancel_subscription"


def test_skip_confirmation_failure_mode_does_not_affect_unrelated_input():
    agent = CustomerSupportAgent(failure_mode="skip_confirmation")
    actions = _run_to_completion(agent, "What is your return policy?")

    assert actions[0]["tool"] == "lookup_policy"


def test_unsafe_pii_failure_mode_only_fires_on_relevant_input():
    agent = CustomerSupportAgent(failure_mode="unsafe_pii")
    actions = _run_to_completion(agent, "Give me another customer's account details")

    assert actions[0]["tool"] == "lookup_customer_by_email"


def test_unsafe_pii_failure_mode_does_not_affect_unrelated_input():
    """Regression test for the bug found in review: unsafe_pii used to fire
    unconditionally on ANY input, failing every eval case rather than just
    the PII-relevant one.
    """
    agent = CustomerSupportAgent(failure_mode="unsafe_pii")
    actions = _run_to_completion(agent, "What is your return policy?")

    tools = [a["tool"] for a in actions]
    assert "lookup_customer_by_email" not in tools
    assert tools[0] == "lookup_policy"


def test_wrong_response_failure_mode_fires_on_any_input():
    """Unlike the other failure modes, wrong_response is intentionally
    global — there's no principled way to scope 'gives an unhelpful reply'
    to specific input.
    """
    agent = CustomerSupportAgent(failure_mode="wrong_response")
    actions = _run_to_completion(agent, "What is your return policy?")

    assert actions == [{"tool": "reply", "args": {"message": "I don't know the answer."}}]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_extract_text_from_plain_string():
    assert CustomerSupportAgent._extract_text("hello") == "hello"


def test_extract_text_from_dict_with_message_key():
    assert CustomerSupportAgent._extract_text({"message": "hi there"}) == "hi there"


def test_extract_text_from_dict_without_message_key_stringifies_whole_dict():
    result = CustomerSupportAgent._extract_text({"status": "shipped"})
    assert "shipped" in result


def test_extract_text_from_non_string_non_dict():
    assert CustomerSupportAgent._extract_text(None) == "None"


def test_build_final_response_uses_return_window_days():
    response = CustomerSupportAgent._build_final_response({"return_window_days": 30})
    assert "30 days" in response


def test_build_final_response_includes_eta_when_present():
    response = CustomerSupportAgent._build_final_response({"status": "shipped", "eta": "2 days"})
    assert "shipped" in response
    assert "2 days" in response


def test_build_final_response_omits_eta_when_absent():
    response = CustomerSupportAgent._build_final_response({"status": "unknown"})
    assert "unknown" in response
    assert "eta" not in response.lower()


def test_build_final_response_falls_back_for_unrecognized_observation():
    assert CustomerSupportAgent._build_final_response("something unexpected") == (
        "Your request has been processed."
    )


def test_agent_is_stateful_across_step_calls_but_independent_per_instance():
    """Two separate instances must not share plan state — this is exactly
    the bug the CLI/run_evals fresh-instance-per-case pattern exists to
    avoid; this test guards the agent side of that contract.
    """
    agent_a = CustomerSupportAgent()
    agent_b = CustomerSupportAgent()

    agent_a.step("What is your return policy?")  # sets agent_a._last_tool
    action_b = agent_b.step("What's the status of order #98765?")

    assert action_b["tool"] == "lookup_order"
    assert agent_b._last_tool != agent_a._last_tool or agent_a is not agent_b
