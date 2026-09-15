"""Command-line interface for AgentGuard."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from agentguard.contracts.schema import Contract
from agentguard.core.harness import Harness
from agentguard.core.runner import Scenario

app = typer.Typer(help="AgentGuard: testing and chaos engineering for AI agents")
console = Console()


def _load_agent(dotted_path: str) -> Any:
    """Load an agent class/instance from a 'module.submodule:ClassName' path."""
    module_path, _, attr = dotted_path.partition(":")
    module = importlib.import_module(module_path)
    obj = getattr(module, attr)
    return obj() if isinstance(obj, type) else obj


@app.command()
def run(
    eval_suite: Path = typer.Argument(..., help="Path to an eval suite JSON file"),
    agent: str = typer.Option(
        ..., help="Dotted path to an agent, e.g. 'agents.customer_support.agent:CustomerSupportAgent'"
    ),
):
    """Run every case in an eval suite through the Harness and report PASS/FAIL."""
    suite = json.loads(eval_suite.read_text())
    cases = suite.get("cases", [])
    console.print(
        f"Loaded eval suite [bold]{eval_suite.name}[/bold] with {len(cases)} case(s) "
        f"for agent '{agent}'."
    )

    agent_obj = _load_agent(agent)

    # A no-op "reply" tool so the shipped example agent (which only emits
    # {"tool": "reply", ...}) can run end-to-end. Real agents with real
    # tools (lookup_order, issue_refund, etc.) need those passed in here —
    # this CLI command doesn't yet have a way to load tool implementations
    # from the eval suite, only the agent and its contract.
    default_tools = {"reply": lambda **kwargs: kwargs}

    table = Table(title=f"AgentGuard — {suite.get('suite_name', eval_suite.stem)}")
    table.add_column("Case")
    table.add_column("Verdict")
    table.add_column("Reason")

    all_passed = True
    for case in cases:
        contract = Contract(**case["contract"]) if "contract" in case else None
        harness = Harness(agent=agent_obj, tools=default_tools, contract=contract)
        result = harness.run(Scenario(name=case["id"], initial_input=case["input"]))
        all_passed &= result.passed

        reason = (
            result.contract_result.reason
            if result.contract_result and not result.contract_result.passed
            else "-"
        )
        style = "green" if result.passed else "red"
        table.add_row(case["id"], f"[{style}]{result.verdict}[/{style}]", reason)

    console.print(table)
    if not all_passed:
        raise typer.Exit(code=1)


@app.command()
def report(results_dir: Path = typer.Argument(..., help="Directory of run results")):
    """Summarize results from a previous run."""
    table = Table(title="AgentGuard Run Summary")
    table.add_column("Scenario")
    table.add_column("Passed")
    console.print(table)
    console.print(f"[yellow]No results loader implemented yet for {results_dir}[/yellow]")


if __name__ == "__main__":
    app()
