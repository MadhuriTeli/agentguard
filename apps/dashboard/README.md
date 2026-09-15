# AgentGuard Dashboard

Web dashboard for viewing evaluation runs, trajectories, and regression
reports produced by the AgentGuard API.

Scaffold this with your frontend framework of choice (e.g. Next.js, Vite +
React). Point it at the API's base URL via `NEXT_PUBLIC_API_URL` (see
`docker-compose.yml`).

Suggested pages:
- `/runs` — list of evaluation runs
- `/runs/[id]` — trajectory viewer + per-eval verdicts
- `/regressions` — baseline vs candidate diff view
