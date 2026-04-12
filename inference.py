"""
inference.py — SQL Query Optimizer
Uses validator-injected API_BASE_URL and API_KEY for LLM proxy.
"""

import os
import sys
import json
import time
import argparse
import requests
from openai import OpenAI

DEFAULT_BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://localhost:7860")
MAX_STEPS = 6

GREEDY_ORDER = [
    "eliminate_subquery",
    "push_predicate",
    "add_index_hint",
    "reorder_joins",
    "remove_redundant_columns",
]

SYSTEM_PROMPT = """You are a SQL query optimization expert agent.
Pick ONE optimization action per step to reduce query cost.

Available actions:
- eliminate_subquery
- push_predicate
- add_index_hint
- reorder_joins
- remove_redundant_columns
- no_op

Respond with ONLY a JSON object like:
{"optimization_type": "eliminate_subquery", "target_table": "", "hint_value": ""}"""


def get_llm_client():
    """Use validator-injected API_BASE_URL and API_KEY if available."""
    api_base = os.environ.get("API_BASE_URL", None)
    api_key  = os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "dummy-key"))
    if api_base:
        print(f"Using LLM proxy: {api_base}", flush=True)
        return OpenAI(base_url=api_base, api_key=api_key)
    return OpenAI(api_key=api_key)


def ask_llm(client, obs: dict) -> dict:
    try:
        user_msg = f"""Step: {obs.get('step_count',0)}/{MAX_STEPS}
Cost: {obs.get('estimated_cost')} (original: {obs.get('original_cost')})
Applied: {obs.get('optimization_history',[])}
Schema: {obs.get('schema_context','')}
Query: {obs.get('current_query','')}
Pick next action. Reply ONLY with JSON."""

        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        action = json.loads(raw)
        valid = ["eliminate_subquery","push_predicate","add_index_hint","reorder_joins","remove_redundant_columns","no_op"]
        if action.get("optimization_type") not in valid:
            action["optimization_type"] = "no_op"
        return action
    except Exception as e:
        print(f"  LLM error: {e} — using greedy fallback", flush=True)
        # Greedy fallback
        history = obs.get("optimization_history", [])
        for opt in GREEDY_ORDER:
            if opt not in history:
                return {"optimization_type": opt, "target_table": "", "hint_value": ""}
        return {"optimization_type": "no_op", "target_table": "", "hint_value": ""}


def wait_for_server(base_url: str, retries: int = 20, delay: int = 5) -> bool:
    print(f"Waiting for server at {base_url} ...", flush=True)
    for i in range(retries):
        for path in ["/health", "/docs", "/tasks"]:
            try:
                r = requests.get(f"{base_url}{path}", timeout=8)
                if r.status_code == 200:
                    print(f"Server ready (attempt {i+1})", flush=True)
                    return True
            except Exception:
                pass
        print(f"  attempt {i+1}/{retries} — retrying in {delay}s ...", flush=True)
        time.sleep(delay)
    return False


def run_episode(base_url: str, task: dict, client) -> dict:
    task_id    = task["task_id"]
    difficulty = task["difficulty"]

    print(f"[START] task={task_id}", flush=True)

    r = requests.post(f"{base_url}/reset", json={}, timeout=30)
    r.raise_for_status()
    data = r.json()
    obs  = data.get("observation", data)

    total_reward = 0.0
    done  = False
    steps = 0
    applied = []

    while not done and steps < MAX_STEPS:
        action = ask_llm(client, obs)
        action_type = action.get("optimization_type", "no_op")

        resp = requests.post(
            f"{base_url}/step",
            json={"action": {
                "optimization_type": action_type,
                "target_table": action.get("target_table", ""),
                "hint_value":   action.get("hint_value", ""),
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

        print(f"[STEP] step={steps} action={action_type} reward={round(reward,4)}", flush=True)

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

    print(f"[END] task={task_id} score={round(score,4)} steps={steps}", flush=True)

    return {
        "task_id":            task_id,
        "difficulty":         difficulty,
        "score":              score,
        "total_reward":       round(total_reward, 2),
        "steps":              steps,
        "cost_reduction_pct": graded.get("cost_reduction_pct", 0),
        "optimizations":      applied,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    print(f"SQL Query Optimizer - Inference", flush=True)
    print(f"Server: {base_url}", flush=True)
    print(f"API_BASE_URL: {os.environ.get('API_BASE_URL', 'not set')}", flush=True)

    if not wait_for_server(base_url):
        print("ERROR: Server not reachable", flush=True)
        sys.exit(1)

    # Init LLM client with validator proxy
    client = get_llm_client()

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
            result = run_episode(base_url, task, client)
            results.append(result)
        except Exception as e:
            task_id = task.get("task_id", "unknown")
            print(f"[START] task={task_id}", flush=True)
            print(f"[STEP] step=1 action=no_op reward=0.0", flush=True)
            print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
            results.append({"task_id": task_id, "difficulty": task.get("difficulty",""), "score": 0.0, "error": str(e)})

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"average_score={round(avg,4)}", flush=True)

    with open("inference_scores.json", "w") as f:
        json.dump({"agent": "llm+greedy", "average_score": round(avg,4), "tasks": results}, f, indent=2)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
