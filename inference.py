"""
inference.py — SQL Query Optimizer
Required at repo root by OpenEnv Hackathon Phase 2 validator.

Prints structured [START]/[STEP]/[END] blocks to stdout as required.
"""

import os
import sys
import json
import time
import argparse
import requests

DEFAULT_BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://localhost:7860")
MAX_STEPS = 6

GREEDY_ORDER = [
    "eliminate_subquery",
    "push_predicate",
    "add_index_hint",
    "reorder_joins",
    "remove_redundant_columns",
]


def wait_for_server(base_url: str, retries: int = 20, delay: int = 5) -> bool:
    print(f"Waiting for server at {base_url} ...", flush=True)
    for i in range(retries):
        for path in ["/health", "/docs", "/tasks"]:
            try:
                r = requests.get(f"{base_url}{path}", timeout=8)
                if r.status_code == 200:
                    print(f"Server ready via {path} (attempt {i+1})", flush=True)
                    return True
            except Exception:
                pass
        print(f"  attempt {i+1}/{retries} — retrying in {delay}s ...", flush=True)
        time.sleep(delay)
    return False


def run_episode(base_url: str, task: dict) -> dict:
    task_id    = task["task_id"]
    difficulty = task["difficulty"]

    # ── REQUIRED: [START] block ───────────────────────────────────────────────
    print(f"[START] task={task_id}", flush=True)

    # Reset
    r = requests.post(f"{base_url}/reset", json={}, timeout=30)
    r.raise_for_status()
    data = r.json()
    obs  = data.get("observation", data)

    total_reward = 0.0
    done         = False
    steps        = 0
    tried        = set()
    applied      = []

    while not done and steps < MAX_STEPS:
        action_type = "no_op"
        for opt in GREEDY_ORDER:
            if opt not in tried:
                action_type = opt
                tried.add(opt)
                break

        resp = requests.post(
            f"{base_url}/step",
            json={"action": {
                "optimization_type": action_type,
                "target_table": "",
                "hint_value": "",
            }},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        reward        = result.get("reward", 0.0)
        done          = result.get("done", False)
        obs           = result.get("observation", obs)
        total_reward += reward
        steps        += 1

        if reward > 0:
            applied.append(action_type)

        # ── REQUIRED: [STEP] block ────────────────────────────────────────────
        print(f"[STEP] step={steps} action={action_type} reward={round(reward, 4)}", flush=True)

    # Grade via /grader
    grade = requests.post(f"{base_url}/grader", json={
        "original_cost":         obs.get("original_cost", 0),
        "final_cost":            obs.get("estimated_cost", 0),
        "optimizations_applied": obs.get("optimization_history", applied),
        "steps_taken":           obs.get("step_count", steps),
        "is_valid_sql":          obs.get("is_valid_sql", True),
    }, timeout=30)
    grade.raise_for_status()
    graded = grade.json()
    score  = graded.get("score", 0.0)

    # ── REQUIRED: [END] block ─────────────────────────────────────────────────
    print(f"[END] task={task_id} score={round(score, 4)} steps={steps}", flush=True)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    print(f"SQL Query Optimizer - Inference", flush=True)
    print(f"Server: {base_url}", flush=True)

    # Wait for server to be healthy
    if not wait_for_server(base_url):
        print(f"ERROR: Server not reachable at {base_url}", flush=True)
        sys.exit(1)

    # Fetch tasks
    try:
        r = requests.get(f"{base_url}/tasks", timeout=15)
        r.raise_for_status()
        tasks = r.json().get("tasks", [])
        if not tasks:
            raise ValueError("Empty task list")
    except Exception as e:
        print(f"ERROR fetching /tasks: {e}", flush=True)
        sys.exit(1)

    print(f"Found {len(tasks)} tasks", flush=True)

    results = []
    for task in tasks:
        try:
            result = run_episode(base_url, task)
            results.append(result)
        except Exception as e:
            task_id = task.get("task_id", "unknown")
            print(f"[START] task={task_id}", flush=True)
            print(f"[STEP] step=1 action=no_op reward=0.0", flush=True)
            print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
            results.append({
                "task_id":    task_id,
                "difficulty": task.get("difficulty", "unknown"),
                "score":      0.0,
                "error":      str(e),
            })

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"average_score={round(avg, 4)}", flush=True)

    # Save scores to file
    with open("inference_scores.json", "w") as f:
        json.dump({
            "agent":         "greedy",
            "base_url":      base_url,
            "average_score": round(avg, 4),
            "tasks":         results,
        }, f, indent=2)

    print("Done. Scores saved to inference_scores.json", flush=True)


if __name__ == "__main__":
    main()