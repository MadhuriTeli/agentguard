"""Run AgentGuard evaluation suites from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the repository root importable when this script is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agentguard.contracts.schema import Contract
from agentguard.core.harness import Harness
from agentguard.core.runner import Scenario
from agentguard.judges.rule_judge import RuleJudge
from agentguard.judges.trajectory_judge import TrajectoryJudge
from agents.customer_support.agent import CustomerSupportAgent
from agents.customer_support.tools import TOOLS


def load_eval_suites(agent_name: str) -> list[dict[str, Any]]:
    """Load all JSON evaluation suites for an agent."""
    eval_dir = REPO_ROOT / "evals" / agent_name

    if not eval_dir.exists():
        raise FileNotFoundError(
            f"Evaluation directory not found: {eval_dir}"
        )

    suites = []

    for path in sorted(eval_dir.glob("*.json")):
        suites.append(json.loads(path.read_text()))

    return suites


def build_agent(agent_name: str):
    """Create the agent under evaluation."""
    if agent_name == "customer_support":
        return CustomerSupportAgent()

    raise ValueError(f"Unknown agent: {agent_name}")


def run_case(agent_name: str, case: dict[str, Any]):
    """Execute one evaluation case through AgentGuard."""

    agent = build_agent(agent_name)

    contract_data = case.get("contract")

    contract = (
        Contract(**contract_data)
        if contract_data
        else None
    )

    evaluators = [
        RuleJudge(),
    ]

    expect = case.get("expect", {})

    if expect:
        evaluators.append(
            TrajectoryJudge(
                expected_tool_calls=expect.get("tool_calls"),
                final_response_contains=expect.get(
                    "final_response_contains"
                ),
            )
        )

    harness = Harness(
        agent=agent,
        tools=TOOLS,
        contract=contract,
        evaluators=evaluators,
    )

    scenario = Scenario(
        name=case["id"],
        initial_input=case["input"],
        max_steps=10,
        is_done=lambda trajectory: any(
            isinstance(step.content, dict)
            and step.content.get("tool") == "reply"
            for step in trajectory.tool_calls()
        ),
    )

    return harness.run(scenario)


def print_result(case: dict[str, Any], result) -> None:
    """Print one case result."""

    status = "PASS" if result.passed else "FAIL"

    print(f"\n[{status}] {case['id']}")
    print(f"  Input: {case['input']}")
    print(f"  Verdict: {result.verdict}")

    if result.run_result:
        for evaluation in result.run_result.eval_results:
            print(
                f"  {evaluation.name}: "
                f"{evaluation.verdict.value} — "
                f"{evaluation.reason}"
            )

    if result.contract_result:
        print(
            f"  contract: "
            f"{result.contract_result.verdict.value} — "
            f"{result.contract_result.reason}"
        )

    print(
        f"  steps={result.metrics.num_steps} "
        f"tool_calls={result.metrics.num_tool_calls} "
        f"duration={result.metrics.duration_seconds:.4f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AgentGuard evaluation suites."
    )

    parser.add_argument(
        "--agent",
        required=True,
        help="Agent name, e.g. customer_support",
    )

    args = parser.parse_args()

    suites = load_eval_suites(args.agent)

    total_cases = sum(
        len(suite.get("cases", []))
        for suite in suites
    )

    print(
        f"Found {len(suites)} suite(s), "
        f"{total_cases} case(s) for agent '{args.agent}':"
    )

    for suite in suites:
        print(
            f"  - {suite['suite_name']}: "
            f"{len(suite.get('cases', []))} case(s)"
        )

    passed = 0
    failed = 0

    for suite in suites:
        print(f"\n{'=' * 70}")
        print(suite["suite_name"])
        print(suite.get("description", ""))
        print(f"{'=' * 70}")

        for case in suite.get("cases", []):
            result = run_case(args.agent, case)
            print_result(case, result)

            if result.passed:
                passed += 1
            else:
                failed += 1

    print(f"\n{'=' * 70}")
    print("AGENTGUARD EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total:  {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if passed + failed:
        print(
            f"Pass rate: "
            f"{(passed / (passed + failed)) * 100:.1f}%"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
