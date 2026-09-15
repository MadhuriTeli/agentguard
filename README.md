# AgentGuard

AgentGuard is an AI agent evaluation and reliability platform. It lets you
benchmark agents, enforce behavioral contracts, inject chaos to test
resilience, trace and replay failures, and gate CI on regressions.

## Architecture

```
AgentGuard
       AI Agent Evaluation & Reliability Platform
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
     AgentBench          Agent Contracts      Agent Chaos
     Evaluation          Safety/Policy        Resilience
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                     Trajectory Tracing
                              │
                              ▼
                       Failure Replay
                              │
                              ▼
                    Regression Detection
                              │
                              ▼
                       GitHub CI Gate
```

| Layer | What it does | Implementation |
|---|---|---|
| AgentBench Evaluation | Scores agent behavior against eval suites | `agentguard/core/evaluator.py`, `agentguard/judges/` |
| Agent Contracts | Declarative safety/policy constraints on tool use | `agentguard/contracts/` |
| Agent Chaos | Fault injection to test resilience | `agentguard/chaos/` |
| Trajectory Tracing | Records every step of a run | `agentguard/core/trajectory.py` |
| Failure Replay | Deterministically reproduces a failing run against a patched agent | `agentguard/core/replay.py` |
| Regression Detection | Diffs baseline vs candidate results | `agentguard/regression/comparator.py` |
| GitHub CI Gate | Fails the build on regressions | `.github/workflows/agentguard-ci.yml` |

### The Harness

`agentguard.core.harness.Harness` is the orchestration layer most users
actually want — it wraps an agent's tools and runs one call through the
whole pipeline in order:

```
Agent
  ↓
AgentGuard Harness
  ├── intercept tools       — agent calls wrapped versions of your tool fns
  ├── enforce contracts     — forbidden tools / max_calls blocked live, mid-run
  ├── record trajectory     — every call, result, fault, and violation logged
  ├── inject failures       — ChaosEngine mutates tool results before return
  ├── evaluate result       — ContractValidator + your judges score the run
  ├── calculate metrics     — duration, call counts, fault/violation counts
  └── detect regressions    — diff against an optional baseline RunResult
  ↓
PASS / FAIL
```

```python
from agentguard.core.harness import Harness
from agentguard.core.runner import Scenario
from agentguard.contracts.schema import Contract

harness = Harness(
    agent=my_agent,
    tools={"lookup_order": lookup_order, "issue_refund": issue_refund},
    contract=Contract(name="no_unverified_refunds", forbidden_tools=["issue_refund"]),
    evaluators=[RuleJudge(forbidden_patterns=[r"\bguarantee\b"])],
    baseline=previous_run_result,  # optional
)

result = harness.run(Scenario(name="angry_customer", initial_input="Refund me $10k now"))
print(result.verdict)     # "PASS" or "FAIL"
print(result.summary())   # full breakdown for logging / CI artifacts
```

Two enforcement points worth noting: violations that are knowable *per
call* (a forbidden tool, an over-quota tool) are blocked live, inside the
intercepted tool wrapper, before the underlying tool ever runs. Violations
only knowable at the *end* of a run (a required tool that was never called)
are still caught by `ContractValidator` after the trajectory finishes. Both
feed into the same final verdict.

The three top boxes (AgentBench, Contracts, Chaos) all run against the same
agent and feed into one shared Trajectory. From there the pipeline is
linear: a failing trajectory can be replayed for debugging, then
`RegressionComparator` decides whether a candidate is safe to merge, and the
CI workflow enforces that decision as a merge gate.

## Project Layout

```
agentguard/
├── apps/
│   ├── api/           # Backend API service
│   └── dashboard/     # Web dashboard for viewing results
│
├── agentguard/         # Core Python package
│   ├── core/           # Runner, trajectory, evaluator, replay, results
│   ├── contracts/      # Schema + validation for agent behavior contracts
│   ├── chaos/          # Fault injection engine
│   ├── judges/         # LLM-based and rule-based judges
│   └── regression/     # Regression comparison across runs
│
├── agents/             # Example / target agents under test
│   └── customer_support/
│
├── evals/              # Evaluation suites (JSON specs)
│   └── customer_support/
│
├── .github/workflows/   # CI gate (runs evals, fails build on regressions)
├── tests/               # Unit and integration tests
├── scripts/             # Utility / dev scripts
├── pyproject.toml
├── README.md
└── docker-compose.yml
```

## Getting Started

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Start the full stack (API + dashboard)
docker-compose up
```

## Core Concepts

- **Runner** — executes an agent against a scenario and records a trajectory.
- **Trajectory** — the full sequence of agent actions, tool calls, and
  observations for a single run.
- **Contract** — a declarative spec of expected agent behavior (allowed
  tools, required steps, forbidden actions, etc.).
- **Judge** — scores a trajectory, either via deterministic rules or an LLM.
- **Chaos Engine** — injects faults (latency, tool failures, malformed
  responses) into a run to test agent robustness.
- **Replay Session** — deterministically re-runs a previously recorded
  failing trajectory (same inputs, same fault schedule) against a patched
  agent, to confirm whether a fix actually resolves the failure.
- **Regression Comparator** — diffs evaluation results across two runs or
  versions to catch behavioral regressions.

## License

TBD
