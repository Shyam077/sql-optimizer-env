"""
baseline_inference.py
─────────────────────
Baseline inference script for the SQL Query Optimizer OpenEnv environment.

Runs a greedy LLM agent (via OpenAI-compatible API) against all 3 tasks
and produces reproducible scores.

Usage:
    export OPENAI_API_KEY=your_key_here
    python baseline_inference.py

    # Against a live HF Space:
    python baseline_inference.py --base-url https://Shyamj0799-sql-optimizer-env.hf.space

    # Against local server:
    python baseline_inference.py --base-url http://localhost:8000
"""

import os
import sys
import json
import argparse
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8000"
MODEL            = "gpt-4o-mini"           # cheap + fast for baseline
MAX_STEPS        = 6

VALID_ACTIONS = [
    "eliminate_subquery",
    "push_predicate",
    "add_index_hint",
    "reorder_joins",
    "remove_redundant_columns",
    "no_op",
]

SYSTEM_PROMPT = """You are a SQL query optimization expert agent.

You will be given a slow SQL query, its database schema, and current estimated cost.
Your job is to pick ONE optimization action per step to reduce the query cost.

Available actions:
- eliminate_subquery     : Replace correlated subqueries with CTEs or JOINs (+40 pts)
- push_predicate         : Push WHERE filters earlier in the query plan (+25 pts)
- add_index_hint         : Suggest an index on key join/filter columns (+20 pts)
- reorder_joins          : Convert implicit comma joins to explicit INNER JOINs (+15 pts)
- remove_redundant_columns: Narrow SELECT list to required columns only (+5 pts)
- no_op                  : Do nothing (−1 pts, avoid this)

Rules:
- Pick each action at most once per episode (duplicates give −2 penalty)
- Never pick an action not in the list above (gives −10 and ends episode)
- You want to maximize total cost reduction in as few steps as possible

Respond with ONLY a JSON object like:
{"optimization_type": "eliminate_subquery", "target_table": "", "hint_value": ""}
"""


def ask_llm(client: OpenAI, obs: dict) -> dict:
    """Ask the LLM what action to take given the current observation."""
    user_msg = f"""Current state:
- Step: {obs.get('step_count', 0)} / {MAX_STEPS}
- Estimated cost: {obs.get('estimated_cost')} (original: {obs.get('original_cost')})
- Optimizations already applied: {obs.get('optimization_history', [])}

Schema:
{obs.get('schema_context', '')}

Current query:
{obs.get('current_query', '')}

Pick your next optimization action. Reply with ONLY a JSON object."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    action = json.loads(raw)

    # Validate action is in allowed list
    if action.get("optimization_type") not in VALID_ACTIONS:
        print(f"  ⚠ LLM returned invalid action '{action.get('optimization_type')}', defaulting to no_op")
        action["optimization_type"] = "no_op"

    return action


def run_episode(base_url: str, task: dict, client: OpenAI) -> dict:
    """Run one full episode for a given task using the LLM agent."""
    task_id = task["task_id"]
    print(f"\n{'─'*55}")
    print(f"  Task: {task_id} ({task['difficulty'].upper()})")
    print(f"  {task['description'][:60]}...")
    print(f"{'─'*55}")

    # Reset environment
    reset_resp = requests.post(f"{base_url}/reset", json={}, timeout=30)
    reset_resp.raise_for_status()
    obs = reset_resp.json().get("observation", reset_resp.json())

    total_reward = 0.0
    done         = False
    steps        = 0

    while not done and steps < MAX_STEPS:
        action = ask_llm(client, obs)
        print(f"  Step {steps+1}: {action['optimization_type']}", end="")

        step_resp = requests.post(
            f"{base_url}/step",
            json={"action": action},
            timeout=30,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        reward = result.get("reward", 0.0)
        done   = result.get("done", False)
        obs    = result.get("observation", obs)

        total_reward += reward
        steps        += 1
        print(f"  → reward: {reward:+.1f}  cost: {obs.get('estimated_cost', '?')}")

    # Call /grader to get official score
    grade_resp = requests.post(f"{base_url}/grader", json={
        "original_cost":        obs.get("original_cost", 0),
        "final_cost":           obs.get("estimated_cost", 0),
        "optimizations_applied": obs.get("optimization_history", []),
        "steps_taken":          obs.get("step_count", steps),
        "is_valid_sql":         obs.get("is_valid_sql", True),
    }, timeout=30)
    grade_resp.raise_for_status()
    grader_result = grade_resp.json()

    score = grader_result.get("score", 0.0)
    print(f"\n  ✅ Score: {score:.4f}  |  Total reward: {total_reward:+.1f}")
    print(f"     Cost reduction: {grader_result.get('cost_reduction_pct', 0):.1f}%")
    print(f"     Optimizations: {obs.get('optimization_history', [])}")

    return {
        "task_id":            task_id,
        "difficulty":         task["difficulty"],
        "score":              score,
        "total_reward":       round(total_reward, 2),
        "steps":              steps,
        "breakdown":          grader_result.get("breakdown", {}),
        "cost_reduction_pct": grader_result.get("cost_reduction_pct", 0),
        "optimizations":      obs.get("optimization_history", []),
    }


def main():
    parser = argparse.ArgumentParser(description="SQL Optimizer baseline inference script")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Environment server URL")
    parser.add_argument("--model",    default=MODEL,            help="OpenAI model to use")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set.")
        print("   export OPENAI_API_KEY=your_key_here")
        sys.exit(1)

    client   = OpenAI(api_key=api_key)
    base_url = args.base_url.rstrip("/")

    print(f"\n{'='*55}")
    print(f"  SQL Query Optimizer — Baseline Inference")
    print(f"  Server:  {base_url}")
    print(f"  Model:   {args.model}")
    print(f"{'='*55}")

    # Fetch task list
    tasks_resp = requests.get(f"{base_url}/tasks", timeout=15)
    tasks_resp.raise_for_status()
    tasks = tasks_resp.json().get("tasks", [])
    print(f"\n  Found {len(tasks)} tasks")

    # Run each task
    all_results = []
    for task in tasks:
        result = run_episode(base_url, task, client)
        all_results.append(result)

    # Summary
    avg_score = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n{'='*55}")
    print(f"  BASELINE RESULTS")
    print(f"{'='*55}")
    for r in all_results:
        print(f"  {r['task_id']:25s}  score={r['score']:.4f}  ({r['difficulty']})")
    print(f"  {'─'*45}")
    print(f"  Average score: {avg_score:.4f}")
    print(f"{'='*55}\n")

    # Write results to file
    output = {
        "model":         args.model,
        "base_url":      base_url,
        "average_score": round(avg_score, 4),
        "tasks":         all_results,
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to baseline_scores.json")

    return output


if __name__ == "__main__":
    main()
