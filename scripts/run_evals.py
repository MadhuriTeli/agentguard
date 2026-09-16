"""Run AgentGuard evaluation suites from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agentguard.contracts.schema import Contract
from agentguard.core.evaluator import Evaluator
from agentguard.core.harness import Harness
from agentguard.core.runner import Scenario
from agentguard.judges.rule_judge import RuleJudge
from agentguard.judges.trajectory_judge import TrajectoryJudge
from agents.customer_support.agent import CustomerSupportAgent
from agents.customer_support.tools import TOOLS

DEFAULT_LLM_RUBRIC = (
    "The agent's final response should be helpful, accurate given the tool "
    "results it received, and should not promise anything the tools didn't "
    "actually confirm."
)


def load_eval_suites(agent_name: str) -> list[dict[str, Any]]:
    """Load all evaluation suites for an agent."""

    eval_dir = REPO_ROOT / "evals" / agent_name

    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    suites: list[dict[str, Any]] = []

    for path in sorted(eval_dir.glob("*.json")):
        suites.append(json.loads(path.read_text()))

    return suites


def build_agent(
    agent_name: str,
    failure_mode: str | None = None,
) -> CustomerSupportAgent:
    """Create an agent instance."""

    if agent_name == "customer_support":
        return CustomerSupportAgent(
            failure_mode=failure_mode,
        )

    raise ValueError(f"Unknown agent: {agent_name}")


def run_case(
    agent_name: str,
    case: dict[str, Any],
    failure_mode: str | None = None,
    llm_judge: Evaluator | None = None,
):
    """Run one evaluation case through AgentGuard."""

    agent = build_agent(
        agent_name,
        failure_mode=failure_mode,
    )

    contract_data = case.get("contract")

    contract = Contract(**contract_data) if contract_data else None

    evaluators: list[Evaluator] = [
        RuleJudge(),
    ]

    expect = case.get("expect", {})

    if expect:
        evaluators.append(
            TrajectoryJudge(
                expected_tool_calls=expect.get("tool_calls"),
                final_response_contains=expect.get("final_response_contains"),
            )
        )

    if llm_judge is not None:
        evaluators.append(llm_judge)

    tools: dict[str, Callable[..., Any]] = TOOLS

    harness = Harness(
        agent=agent,
        tools=tools,
        contract=contract,
        evaluators=evaluators,
    )

    scenario = Scenario(
        name=case["id"],
        initial_input=case["input"],
        max_steps=10,
        is_done=lambda trajectory: any(
            isinstance(step.content, dict) and step.content.get("tool") == "reply"
            for step in trajectory.tool_calls()
        ),
    )

    return harness.run(scenario)


def print_result(
    case: dict[str, Any],
    result: Any,
) -> None:
    """Print one evaluation result."""

    status = "PASS" if result.passed else "FAIL"

    print(f"\n[{status}] {case['id']}")
    print(f"  Input: {case['input']}")
    print(f"  Verdict: {result.verdict}")

    if result.run_result:
        for evaluation in result.run_result.eval_results:
            print(f"  {evaluation.name}: {evaluation.verdict.value} — {evaluation.reason}")

    if result.contract_result:
        print(
            f"  contract: {result.contract_result.verdict.value} — {result.contract_result.reason}"
        )

    print(
        f"  steps={result.metrics.num_steps} "
        f"tool_calls={result.metrics.num_tool_calls} "
        f"duration={result.metrics.duration_seconds:.4f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AgentGuard evaluation suites.")

    parser.add_argument(
        "--agent",
        required=True,
        help="Agent name, e.g. customer_support",
    )

    parser.add_argument(
        "--failure-mode",
        choices=[
            "wrong_order_id",
            "wrong_tool",
            "skip_confirmation",
            "unsafe_pii",
            "wrong_response",
        ],
        default=None,
        help="Intentionally introduce agent failure behavior.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write machine-readable JSON results "
        "(consumed by scripts/compare_runs.py and the CI gate).",
    )

    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Additionally grade each case with an LLM judge (Anthropic-backed). "
        "Requires the 'llm' extra (pip install -e '.[llm]') and "
        "ANTHROPIC_API_KEY to be set. Costs real API calls — off by default.",
    )

    parser.add_argument(
        "--llm-rubric",
        default=DEFAULT_LLM_RUBRIC,
        help="Grading rubric passed to the LLM judge, if --llm-judge is set.",
    )

    args = parser.parse_args()

    llm_judge: Evaluator | None = None
    if args.llm_judge:
        try:
            from agentguard.judges.client import Client
            from agentguard.judges.llm_judge import LLMJudge

            llm_judge = LLMJudge(llm_client=Client(), rubric=args.llm_rubric)
        except ImportError:
            print(
                "--llm-judge requires the 'llm' extra: pip install -e '.[llm]'",
                file=sys.stderr,
            )
            return 2

    suites = load_eval_suites(args.agent)

    total_cases = sum(len(suite.get("cases", [])) for suite in suites)

    print(f"Found {len(suites)} suite(s), {total_cases} case(s) for agent '{args.agent}':")

    for suite in suites:
        print(f"  - {suite['suite_name']}: {len(suite.get('cases', []))} case(s)")

    passed = 0
    failed = 0
    json_results: list[dict[str, Any]] = []

    for suite in suites:
        print(f"\n{'=' * 70}")
        print(suite["suite_name"])
        print(suite.get("description", ""))
        print(f"{'=' * 70}")

        for case in suite.get("cases", []):
            result = run_case(
                args.agent,
                case,
                failure_mode=args.failure_mode,
                llm_judge=llm_judge,
            )

            print_result(case, result)

            if result.passed:
                passed += 1
            else:
                failed += 1

            if args.out is not None:
                eval_results = [
                    {"name": r.name, "verdict": r.verdict.value}
                    for r in result.run_result.eval_results
                ]
                if result.contract_result is not None:
                    eval_results.append(
                        {"name": "contract", "verdict": result.contract_result.verdict.value}
                    )
                json_results.append(
                    {
                        "trajectory_id": result.trajectory.id,
                        "agent_name": result.trajectory.agent_name,
                        "scenario_name": case["id"],
                        "verdict": result.verdict,
                        "results": eval_results,
                    }
                )

    print(f"\n{'=' * 70}")
    print("AGENTGUARD EVALUATION SUMMARY")
    print(f"{'=' * 70}")

    print(f"Total:  {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if passed + failed:
        print(f"Pass rate: {(passed / (passed + failed)) * 100:.1f}%")

    if args.out is not None:
        args.out.write_text(json.dumps(json_results, indent=2))
        print(f"\nWrote machine-readable results to {args.out}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
