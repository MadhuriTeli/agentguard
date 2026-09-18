"""Command-line interface for AgentGuard."""

# ruff: noqa: B008
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
from agentguard.regression.gate import QualityGate, QualityGateConfig

app = typer.Typer(help="AgentGuard: testing and chaos engineering for AI agents")
console = Console()


def _load_agent_factory(dotted_path: str) -> Any:
    """Load an agent from a 'module.submodule:ClassName' path, returning a
    zero-arg factory so each eval case gets a fresh, state-free instance.
    """
    module_path, _, attr = dotted_path.partition(":")
    module = importlib.import_module(module_path)
    obj = getattr(module, attr)
    return obj if isinstance(obj, type) else (lambda: obj)


def _load_tools(dotted_path: str | None) -> dict[str, Any]:
    """Load a tool-name -> callable mapping from a 'module.submodule:DICT_NAME' path."""
    tools: dict[str, Any] = {"reply": lambda **kwargs: kwargs}
    if not dotted_path:
        return tools
    module_path, _, attr = dotted_path.partition(":")
    module = importlib.import_module(module_path)
    tools.update(getattr(module, attr))
    return tools


def _stop_after_reply(trajectory) -> bool:
    """Default is_done for CLI-driven runs: stop once the agent's last
    tool call was `reply`.

    Without this, a Scenario with no is_done runs the full max_steps every
    time — including agents that finish their real work early and then
    just keep getting asked "anything else?" for the remaining steps.
    That doesn't corrupt PASS/FAIL (contract/trajectory checks tolerate
    trailing calls), but it silently inflates the tool-call metrics
    QualityGate and MetricsComparator depend on being meaningful.
    """
    calls = trajectory.tool_calls()
    if not calls:
        return False
    last_action = calls[-1].content
    return isinstance(last_action, dict) and last_action.get("tool") == "reply"


@app.command()
def run(
    eval_suite: Path = typer.Argument(..., help="Path to an eval suite JSON file"),
    agent: str = typer.Option(
        ...,
        help="Dotted path to an agent, e.g. 'agents.customer_support.agent:CustomerSupportAgent'",
    ),
    tools: str | None = typer.Option(
        None,
        help="Dotted path to a tools dict, e.g. 'agents.customer_support.tools:TOOLS'. "
        "A no-op 'reply' tool is always available even if omitted.",
    ),
):
    """Run every case in an eval suite through the Harness and report PASS/FAIL."""
    suite = json.loads(eval_suite.read_text())
    cases = suite.get("cases", [])
    console.print(
        f"Loaded eval suite [bold]{eval_suite.name}[/bold] with {len(cases)} case(s) "
        f"for agent '{agent}'."
    )

    agent_factory = _load_agent_factory(agent)
    default_tools = _load_tools(tools)

    table = Table(title=f"AgentGuard — {suite.get('suite_name', eval_suite.stem)}")
    table.add_column("Case")
    table.add_column("Verdict")
    table.add_column("Reason")

    all_passed = True
    for case in cases:
        contract = Contract(**case["contract"]) if "contract" in case else None
        # Fresh agent instance per case, so stateful agents don't leak
        # plan/step state from one case into the next.
        harness = Harness(agent=agent_factory(), tools=default_tools, contract=contract)
        result = harness.run(
            Scenario(name=case["id"], initial_input=case["input"], is_done=_stop_after_reply)
        )
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


@app.command()
def ci(
    evals_dir: Path = typer.Option(
        ..., help="Directory of eval suite JSON files, e.g. evals/customer_support"
    ),
    agent: str = typer.Option(
        ...,
        help="Dotted path to an agent, e.g. 'agents.customer_support.agent:CustomerSupportAgent'",
    ),
    tools: str | None = typer.Option(
        None, help="Dotted path to a tools dict, e.g. 'agents.customer_support.tools:TOOLS'."
    ),
    min_pass_rate: float = typer.Option(1.0, help="Minimum required pass rate, e.g. 0.95"),
    max_contract_failures: int = typer.Option(0, help="Max tolerated contract failures"),
    max_safety_failures: int = typer.Option(0, help="Max tolerated safety failures"),
):
    """Run every eval suite in a directory and enforce a quality gate.

    NOTE: this compares the candidate run against fixed thresholds
    (QualityGateConfig), not against a saved baseline — MetricsComparator
    supports baseline-vs-candidate metric regression, but the JSON format
    `run_evals.py --out` writes today only persists verdicts, not the
    HarnessMetrics (tool-call counts, duration) a metrics comparison
    needs. Wiring that through is a follow-up, not done here.
    """
    agent_factory = _load_agent_factory(agent)
    default_tools = _load_tools(tools)

    suite_files = sorted(evals_dir.glob("*.json"))
    if not suite_files:
        console.print(f"[red]No eval suite JSON files found in {evals_dir}[/red]")
        raise typer.Exit(code=2)

    runs = []
    table = Table(title="AgentGuard CI")
    table.add_column("Case")
    table.add_column("Verdict")
    table.add_column("Reason")

    for suite_file in suite_files:
        suite = json.loads(suite_file.read_text())
        for case in suite.get("cases", []):
            contract = Contract(**case["contract"]) if "contract" in case else None
            harness = Harness(agent=agent_factory(), tools=default_tools, contract=contract)
            result = harness.run(
                Scenario(name=case["id"], initial_input=case["input"], is_done=_stop_after_reply)
            )
            runs.append(result)

            reason = (
                result.contract_result.reason
                if result.contract_result and not result.contract_result.passed
                else "-"
            )
            style = "green" if result.passed else "red"
            table.add_row(case["id"], f"[{style}]{result.verdict}[/{style}]", reason)

    console.print(table)

    config = QualityGateConfig(
        min_pass_rate=min_pass_rate,
        max_contract_failures=max_contract_failures,
        max_safety_failures=max_safety_failures,
    )
    gate_result = QualityGate().evaluate(runs, config=config)

    metrics_table = Table(title="Quality Gate")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value")
    metrics_table.add_row("Pass rate", f"{gate_result.metrics.pass_rate:.1%}")
    metrics_table.add_row("Contract failures", str(gate_result.metrics.contract_failures))
    metrics_table.add_row("Safety failures", str(gate_result.metrics.safety_failures))
    metrics_table.add_row("Avg tool calls", f"{gate_result.metrics.avg_tool_calls:.1f}")
    console.print(metrics_table)

    if gate_result.passed:
        console.print("[bold green]QUALITY GATE: PASSED[/bold green]")
    else:
        console.print("[bold red]QUALITY GATE: FAILED[/bold red]")
        for reason in gate_result.reasons:
            console.print(f"  [red]- {reason}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
