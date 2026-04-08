"""
inference.py
────────────
Required by the OpenEnv Hackathon Phase 1 validator.
Must exist at the repository root.

Runs a deterministic greedy agent against all 3 tasks and
produces reproducible baseline scores without needing an API key.

Usage:
    # Deterministic greedy (no API key needed):
    python inference.py

    # With LLM agent (requires OPENAI_API_KEY):
    python inference.py --base-url https://Shyamj0799-sql-optimizer-env.hf.space

    # Against local server:
    python inference.py --base-url http://localhost:7860
"""

import os
import sys
import json
import argparse
import requests

DEFAULT_BASE_URL = "http://localhost:7860"
MAX_STEPS = 6

GREEDY_ORDER = [
    "eliminate_subquery",
    "push_predicate",
    "add_index_hint",
    "reorder_joins",
    "remove_redundant_columns",
]


def run_greedy_episode(base_url: str, task: dict) -> dict:
    """Run one episode using a deterministic greedy agent."""
    task_id    = task["task_id"]
    difficulty = task["difficulty"]
    print(f"\n  [{difficulty.upper()}] {task_id}")

    # Reset
    resp = requests.post(f"{base_url}/reset", json={}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    obs  = data.get("observation", data)

    total_reward = 0.0
    done         = False
    steps        = 0
    tried        = set()
    applied      = []

    while not done and steps < MAX_STEPS:
        # Pick next untried greedy action
        action_type = "no_op"
        for opt in GREEDY_ORDER:
            if opt not in tried:
                action_type = opt
                tried.add(opt)
                break

        step_resp = requests.post(
            f"{base_url}/step",
            json={"action": {"optimization_type": action_type, "target_table": "", "hint_value": ""}},
            timeout=30,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        reward       = result.get("reward", 0.0)
        done         = result.get("done", False)
        obs          = result.get("observation", obs)
        total_reward += reward
        steps        += 1

        if reward > 0:
            applied.append(action_type)

        print(f"    step {steps}: {action_type:30s} reward={reward:+.1f}  cost={obs.get('estimated_cost','?')}")

    # Grade via /grader
    grade_resp = requests.post(f"{base_url}/grader", json={
        "original_cost":         obs.get("original_cost", 0),
        "final_cost":            obs.get("estimated_cost", 0),
        "optimizations_applied": obs.get("optimization_history", applied),
        "steps_taken":           obs.get("step_count", steps),
        "is_valid_sql":          obs.get("is_valid_sql", True),
    }, timeout=30)
    grade_resp.raise_for_status()
    graded = grade_resp.json()
    score  = graded.get("score", 0.0)

    print(f"    → score={score:.4f}  cost_reduction={graded.get('cost_reduction_pct',0):.1f}%  optimizations={applied}")

    return {
        "task_id":            task_id,
        "difficulty":         difficulty,
        "score":              score,
        "total_reward":       round(total_reward, 2),
        "steps":              steps,
        "cost_reduction_pct": graded.get("cost_reduction_pct", 0),
        "optimizations":      applied,
        "breakdown":          graded.get("breakdown", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="SQL Optimizer inference script")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Environment server base URL")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    print(f"\n{'='*55}")
    print(f"  SQL Query Optimizer — Inference")
    print(f"  Server : {base_url}")
    print(f"  Agent  : greedy (deterministic, no API key needed)")
    print(f"{'='*55}")

    # Fetch tasks
    try:
        tasks_resp = requests.get(f"{base_url}/tasks", timeout=15)
        tasks_resp.raise_for_status()
        tasks = tasks_resp.json().get("tasks", [])
    except Exception as e:
        print(f"\n❌ Could not reach server at {base_url}: {e}")
        print("   Make sure the server is running before executing inference.py")
        sys.exit(1)

    print(f"\n  Found {len(tasks)} tasks\n")

    results = []
    for task in tasks:
        result = run_greedy_episode(base_url, task)
        results.append(result)

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0

    print(f"\n{'='*55}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*55}")
    for r in results:
        print(f"  {r['task_id']:25s}  score={r['score']:.4f}  ({r['difficulty']})")
    print(f"  {'─'*45}")
    print(f"  Average score : {avg_score:.4f}")
    print(f"{'='*55}\n")

    # Save to file
    output = {
        "agent":         "greedy",
        "base_url":      base_url,
        "average_score": round(avg_score, 4),
        "tasks":         results,
    }
    with open("inference_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Scores saved → inference_scores.json\n")

    return output


if __name__ == "__main__":
    main()
