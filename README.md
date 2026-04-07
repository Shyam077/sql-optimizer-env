---
title: SQL Query Optimizer RL Environment
emoji: 🔵
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - sql
  - query-optimization
---

# SQL Query Optimizer — OpenEnv RL Environment

A Mini-RL environment where an LLM agent learns to iteratively optimize slow SQL queries
by reducing estimated execution cost — step by step.

Built for the **Meta × PyTorch OpenEnv Hackathon 2026**.

---

## The Problem

Slow SQL queries cost companies millions in compute and DBA time. This environment trains
an AI agent to act as an intelligent SQL performance advisor — automatically identifying
and applying the right optimizations in the right order.

---

## Quick Start

```python
from sql_optimizer_env import SQLOptimizerEnv, SQLAction

with SQLOptimizerEnv(base_url="https://Shyamj0799-sql-optimizer-env.hf.space").sync() as env:
    obs = env.reset()
    print(f"Cost: {obs.estimated_cost}")

    result = env.step(SQLAction(optimization_type="eliminate_subquery"))
    print(f"New cost: {result.observation.estimated_cost}  Reward: {result.reward}")
```

---

## Environment Spec

| Property | Value |
|---|---|
| Action space | Discrete — 6 optimization types |
| Observation | Current query, schema, cost estimate, history |
| Reward | Cost units reduced per step (continuous) |
| Max steps | 6 per episode |
| Early stop | Cost drops ≥ 50% below original |
| Scenarios | 3 (e-commerce, HR payroll, insurance claims) |

---

## Action Space

| Action | Reward |
|---|---|
| `eliminate_subquery` | +40 pts |
| `push_predicate` | +25 pts |
| `add_index_hint` | +20 pts |
| `reorder_joins` | +15 pts |
| `remove_redundant_columns` | +5 pts |
| `no_op` | −1 pts |
| Duplicate action | −2 pts |
| Invalid action | −10 pts, episode ends |

---

## Tasks (Easy → Medium → Hard)

### Task 1 — Easy: Single Subquery Elimination
- **Scenario:** E-commerce orders (4 tables, up to 8M rows)
- **Objective:** Eliminate the correlated subquery
- **Baseline cost:** 110 pts
- **Success threshold:** 0.4

### Task 2 — Medium: Multi-Optimization Payroll Query
- **Scenario:** HR payroll (3 tables, up to 1.2M rows)
- **Objective:** Eliminate subquery + push predicates + reorder joins
- **Baseline cost:** 85 pts
- **Success threshold:** 0.6

### Task 3 — Hard: Full Enterprise Insurance Query
- **Scenario:** Insurance claims (4 tables, up to 3M rows)
- **Objective:** 3+ distinct optimizations including IN subquery elimination
- **Baseline cost:** 110 pts
- **Success threshold:** 0.8

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Apply one optimization action |
| `/state` | GET | Get current episode metadata |
| `/tasks` | GET | List all 3 tasks + action schema |
| `/grader` | POST | Score a completed episode (0.0–1.0) |
| `/baseline` | GET | Run greedy baseline agent on all tasks |
| `/web` | GET | Interactive web UI |
| `/docs` | GET | OpenAPI documentation |

---

## Grader

Scores a completed episode 0.0–1.0:

- **Cost reduction (70%)** — how much the estimated cost dropped
- **Optimization diversity (20%)** — how many distinct optimizations were applied
- **Efficiency (10%)** — fewer steps = higher score

---

## Project Structure

```
sql_optimizer_env/
├── __init__.py              ← Exports SQLAction, SQLObservation, SQLOptimizerEnv
├── models.py                ← Pydantic Action + Observation definitions
├── client.py                ← SQLOptimizerEnv (EnvClient — async + sync)
├── baseline_inference.py    ← Baseline LLM agent (OpenAI API)
├── test_sql_env.py          ← 32 unit tests (32/32 passing)
├── openenv.yaml             ← Environment manifest
├── pyproject.toml           ← Dependencies
└── server/
    ├── sql_environment.py   ← Core RL logic (reset, step, reward, SQL rewrites)
    ├── app.py               ← FastAPI server + /tasks /grader /baseline endpoints
    └── Dockerfile           ← Container image
```

---

## Run Baseline Inference

```bash
export OPENAI_API_KEY=your_key_here
python baseline_inference.py --base-url https://Shyamj0799-sql-optimizer-env.hf.space
```

Results are saved to `baseline_scores.json`.

---

## Deploy Locally

```bash
pip install openenv-core fastapi uvicorn
cd sql_optimizer_env
uvicorn server.app:app --reload --port 8000
```

Then open `http://localhost:8000/web`.

---

## License

BSD 3-Clause
