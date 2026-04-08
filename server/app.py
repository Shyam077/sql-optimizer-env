import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.responses import JSONResponse, RedirectResponse
from openenv.core.env_server import create_web_interface_app

from server.sql_environment import (
    SQLOptimizerEnvironment,
    QUERY_SCENARIOS,
    MAX_STEPS,
)
from models import SQLAction, SQLObservation

# ── Create the main OpenEnv app ───────────────────────────────────────────────
# CRITICAL: add_routes=True so /tasks /grader /baseline are registered
# BEFORE the wildcard WebSocket mount — avoids route shadowing.
app = create_web_interface_app(
    SQLOptimizerEnvironment,
    SQLAction,
    SQLObservation,
)

# ── Root → redirect to web UI ─────────────────────────────────────────────────
@app.get("/")
def root():
    return RedirectResponse(url="/web")


# ── /tasks ────────────────────────────────────────────────────────────────────
@app.get("/tasks")
def get_tasks():
    return JSONResponse({
        "tasks": [
            {
                "task_id":           "task_1_easy",
                "name":              "Single Subquery Elimination",
                "difficulty":        "easy",
                "description":       (
                    "Optimize an e-commerce orders query. "
                    "Eliminate the correlated subquery to achieve ≥40% cost reduction."
                ),
                "scenario":          "ecommerce_orders",
                "success_threshold": 0.4,
                "max_steps":         MAX_STEPS,
            },
            {
                "task_id":           "task_2_medium",
                "name":              "Multi-Optimization Payroll Query",
                "difficulty":        "medium",
                "description":       (
                    "Optimize an HR payroll query. "
                    "Apply subquery elimination + predicate pushdown + join reordering."
                ),
                "scenario":          "hr_payroll",
                "success_threshold": 0.6,
                "max_steps":         MAX_STEPS,
            },
            {
                "task_id":           "task_3_hard",
                "name":              "Full Enterprise Insurance Query",
                "difficulty":        "hard",
                "description":       (
                    "Optimize a 4-table insurance claims query. "
                    "Apply ≥3 distinct optimizations to score above 0.8."
                ),
                "scenario":          "insurance_claims",
                "success_threshold": 0.8,
                "max_steps":         MAX_STEPS,
            },
        ],
        "action_schema": {
            "type": "object",
            "properties": {
                "optimization_type": {
                    "type":        "string",
                    "description": "The optimization to apply.",
                    "enum": [
                        "reorder_joins",
                        "add_index_hint",
                        "push_predicate",
                        "eliminate_subquery",
                        "remove_redundant_columns",
                        "no_op",
                    ],
                },
                "target_table": {
                    "type":    "string",
                    "default": "",
                },
                "hint_value": {
                    "type":    "string",
                    "default": "",
                },
            },
            "required": ["optimization_type"],
        },
    })


# ── /grader ───────────────────────────────────────────────────────────────────
@app.post("/grader")
def grade_episode(payload: dict):
    try:
        original_cost  = float(payload.get("original_cost", 0))
        final_cost     = float(payload.get("final_cost", 0))
        optimizations  = payload.get("optimizations_applied", [])
        steps_taken    = int(payload.get("steps_taken", 1))
        is_valid_sql   = bool(payload.get("is_valid_sql", True))

        if not is_valid_sql:
            return JSONResponse({
                "score": 0.0,
                "breakdown": {"cost_reduction": 0.0, "diversity": 0.0, "efficiency": 0.0},
                "reason": "Final query is not valid SQL.",
            })

        if original_cost <= 0:
            return JSONResponse({"score": 0.0, "reason": "Invalid original_cost."})

        # 70% — cost reduction (full marks at 50% reduction)
        reduction_pct = max(0.0, (original_cost - final_cost) / original_cost)
        cost_score    = min(1.0, reduction_pct / 0.5)

        # 20% — diversity (full marks at 3 distinct optimizations)
        unique_opts     = len(set(o for o in optimizations if "(no_effect)" not in str(o)))
        diversity_score = min(1.0, unique_opts / 3.0)

        # 10% — efficiency (fewer steps = better)
        efficiency_score = max(0.0, 1.0 - (max(steps_taken, 1) - 1) / MAX_STEPS)

        score = round(
            0.70 * cost_score +
            0.20 * diversity_score +
            0.10 * efficiency_score,
            4,
        )

        return JSONResponse({
            "score": score,
            "breakdown": {
                "cost_reduction": round(cost_score, 4),
                "diversity":      round(diversity_score, 4),
                "efficiency":     round(efficiency_score, 4),
            },
            "cost_reduction_pct":    round(reduction_pct * 100, 1),
            "optimizations_applied": optimizations,
        })

    except Exception as e:
        return JSONResponse(
            {"score": 0.0, "reason": f"Grader error: {str(e)}"},
            status_code=400,
        )


# ── /baseline ─────────────────────────────────────────────────────────────────
@app.get("/baseline")
def run_baseline():
    """
    Deterministic greedy baseline — no LLM required.
    Always tries optimizations highest-reward-first.
    Results are fully reproducible.
    """
    from uuid import uuid4
    from openenv.core.env_server.types import State

    GREEDY_ORDER = [
        "eliminate_subquery",
        "push_predicate",
        "add_index_hint",
        "reorder_joins",
        "remove_redundant_columns",
    ]

    TASK_SCENARIOS = {
        "task_1_easy":   0,
        "task_2_medium": 1,
        "task_3_hard":   2,
    }

    results = {}

    for task_id, scenario_idx in TASK_SCENARIOS.items():
        # Build a fresh env instance and force the correct scenario
        env = SQLOptimizerEnvironment()
        env._scenario              = QUERY_SCENARIOS[scenario_idx]
        env._cost_factors_remaining = dict(env._scenario["cost_factors"])
        env._original_cost         = sum(env._cost_factors_remaining.values()) + 10.0
        env._current_cost          = env._original_cost
        env._state                 = State(episode_id=str(uuid4()), step_count=0)
        env._history               = []
        env._current_query         = env._scenario["original_query"]

        obs          = env._make_observation(done=False)
        total_reward = 0.0
        applied      = []
        done         = False
        tried        = set()

        for _ in range(MAX_STEPS):
            if done:
                break
            # Pick next untried greedy action
            action_type = "no_op"
            for opt in GREEDY_ORDER:
                if opt not in tried:
                    action_type = opt
                    tried.add(opt)
                    break

            obs, reward, done = env.step(SQLAction(optimization_type=action_type))
            total_reward += reward
            if reward > 0:
                applied.append(action_type)

        reduction_pct    = max(0.0, (obs.original_cost - obs.estimated_cost) / obs.original_cost)
        cost_score       = min(1.0, reduction_pct / 0.5)
        diversity_score  = min(1.0, len(set(applied)) / 3.0)
        efficiency_score = max(0.0, 1.0 - (max(obs.step_count, 1) - 1) / MAX_STEPS)
        score            = round(0.70 * cost_score + 0.20 * diversity_score + 0.10 * efficiency_score, 4)

        results[task_id] = {
            "score":              score,
            "total_reward":       round(total_reward, 2),
            "cost_reduction_pct": round(reduction_pct * 100, 1),
            "optimizations":      applied,
            "steps":              obs.step_count,
            "final_cost":         obs.estimated_cost,
            "original_cost":      obs.original_cost,
        }

    avg = round(sum(r["score"] for r in results.values()) / len(results), 4)

    return JSONResponse({
        "baseline_agent": "greedy (highest-reward-first, deterministic)",
        "average_score":  avg,
        "tasks":          results,
    })
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.responses import JSONResponse, RedirectResponse
from openenv.core.env_server import create_web_interface_app

from server.sql_environment import (
    SQLOptimizerEnvironment,
    QUERY_SCENARIOS,
    MAX_STEPS,
)
from models import SQLAction, SQLObservation

# ── Create the main OpenEnv app ───────────────────────────────────────────────
# CRITICAL: add_routes=True so /tasks /grader /baseline are registered
# BEFORE the wildcard WebSocket mount — avoids route shadowing.
app = create_web_interface_app(
    SQLOptimizerEnvironment,
    SQLAction,
    SQLObservation,
)

# ── Root → redirect to web UI ─────────────────────────────────────────────────
@app.get("/")
def root():
    return RedirectResponse(url="/web")


# ── /tasks ────────────────────────────────────────────────────────────────────
@app.get("/tasks")
def get_tasks():
    return JSONResponse({
        "tasks": [
            {
                "task_id":           "task_1_easy",
                "name":              "Single Subquery Elimination",
                "difficulty":        "easy",
                "description":       (
                    "Optimize an e-commerce orders query. "
                    "Eliminate the correlated subquery to achieve ≥40% cost reduction."
                ),
                "scenario":          "ecommerce_orders",
                "success_threshold": 0.4,
                "max_steps":         MAX_STEPS,
            },
            {
                "task_id":           "task_2_medium",
                "name":              "Multi-Optimization Payroll Query",
                "difficulty":        "medium",
                "description":       (
                    "Optimize an HR payroll query. "
                    "Apply subquery elimination + predicate pushdown + join reordering."
                ),
                "scenario":          "hr_payroll",
                "success_threshold": 0.6,
                "max_steps":         MAX_STEPS,
            },
            {
                "task_id":           "task_3_hard",
                "name":              "Full Enterprise Insurance Query",
                "difficulty":        "hard",
                "description":       (
                    "Optimize a 4-table insurance claims query. "
                    "Apply ≥3 distinct optimizations to score above 0.8."
                ),
                "scenario":          "insurance_claims",
                "success_threshold": 0.8,
                "max_steps":         MAX_STEPS,
            },
        ],
        "action_schema": {
            "type": "object",
            "properties": {
                "optimization_type": {
                    "type":        "string",
                    "description": "The optimization to apply.",
                    "enum": [
                        "reorder_joins",
                        "add_index_hint",
                        "push_predicate",
                        "eliminate_subquery",
                        "remove_redundant_columns",
                        "no_op",
                    ],
                },
                "target_table": {
                    "type":    "string",
                    "default": "",
                },
                "hint_value": {
                    "type":    "string",
                    "default": "",
                },
            },
            "required": ["optimization_type"],
        },
    })


# ── /grader ───────────────────────────────────────────────────────────────────
@app.post("/grader")
def grade_episode(payload: dict):
    try:
        original_cost  = float(payload.get("original_cost", 0))
        final_cost     = float(payload.get("final_cost", 0))
        optimizations  = payload.get("optimizations_applied", [])
        steps_taken    = int(payload.get("steps_taken", 1))
        is_valid_sql   = bool(payload.get("is_valid_sql", True))

        if not is_valid_sql:
            return JSONResponse({
                "score": 0.0,
                "breakdown": {"cost_reduction": 0.0, "diversity": 0.0, "efficiency": 0.0},
                "reason": "Final query is not valid SQL.",
            })

        if original_cost <= 0:
            return JSONResponse({"score": 0.0, "reason": "Invalid original_cost."})

        # 70% — cost reduction (full marks at 50% reduction)
        reduction_pct = max(0.0, (original_cost - final_cost) / original_cost)
        cost_score    = min(1.0, reduction_pct / 0.5)

        # 20% — diversity (full marks at 3 distinct optimizations)
        unique_opts     = len(set(o for o in optimizations if "(no_effect)" not in str(o)))
        diversity_score = min(1.0, unique_opts / 3.0)

        # 10% — efficiency (fewer steps = better)
        efficiency_score = max(0.0, 1.0 - (max(steps_taken, 1) - 1) / MAX_STEPS)

        score = round(
            0.70 * cost_score +
            0.20 * diversity_score +
            0.10 * efficiency_score,
            4,
        )

        return JSONResponse({
            "score": score,
            "breakdown": {
                "cost_reduction": round(cost_score, 4),
                "diversity":      round(diversity_score, 4),
                "efficiency":     round(efficiency_score, 4),
            },
            "cost_reduction_pct":    round(reduction_pct * 100, 1),
            "optimizations_applied": optimizations,
        })

    except Exception as e:
        return JSONResponse(
            {"score": 0.0, "reason": f"Grader error: {str(e)}"},
            status_code=400,
        )


# ── /baseline ─────────────────────────────────────────────────────────────────
@app.get("/baseline")
def run_baseline():
    """
    Deterministic greedy baseline — no LLM required.
    Always tries optimizations highest-reward-first.
    Results are fully reproducible.
    """
    from uuid import uuid4
    from openenv.core.env_server.types import State

    GREEDY_ORDER = [
        "eliminate_subquery",
        "push_predicate",
        "add_index_hint",
        "reorder_joins",
        "remove_redundant_columns",
    ]

    TASK_SCENARIOS = {
        "task_1_easy":   0,
        "task_2_medium": 1,
        "task_3_hard":   2,
    }

    results = {}

    for task_id, scenario_idx in TASK_SCENARIOS.items():
        # Build a fresh env instance and force the correct scenario
        env = SQLOptimizerEnvironment()
        env._scenario              = QUERY_SCENARIOS[scenario_idx]
        env._cost_factors_remaining = dict(env._scenario["cost_factors"])
        env._original_cost         = sum(env._cost_factors_remaining.values()) + 10.0
        env._current_cost          = env._original_cost
        env._state                 = State(episode_id=str(uuid4()), step_count=0)
        env._history               = []
        env._current_query         = env._scenario["original_query"]

        obs          = env._make_observation(done=False)
        total_reward = 0.0
        applied      = []
        done         = False
        tried        = set()

        for _ in range(MAX_STEPS):
            if done:
                break
            # Pick next untried greedy action
            action_type = "no_op"
            for opt in GREEDY_ORDER:
                if opt not in tried:
                    action_type = opt
                    tried.add(opt)
                    break

            obs, reward, done = env.step(SQLAction(optimization_type=action_type))
            total_reward += reward
            if reward > 0:
                applied.append(action_type)

        reduction_pct    = max(0.0, (obs.original_cost - obs.estimated_cost) / obs.original_cost)
        cost_score       = min(1.0, reduction_pct / 0.5)
        diversity_score  = min(1.0, len(set(applied)) / 3.0)
        efficiency_score = max(0.0, 1.0 - (max(obs.step_count, 1) - 1) / MAX_STEPS)
        score            = round(0.70 * cost_score + 0.20 * diversity_score + 0.10 * efficiency_score, 4)

        results[task_id] = {
            "score":              score,
            "total_reward":       round(total_reward, 2),
            "cost_reduction_pct": round(reduction_pct * 100, 1),
            "optimizations":      applied,
            "steps":              obs.step_count,
            "final_cost":         obs.estimated_cost,
            "original_cost":      obs.original_cost,
        }

    avg = round(sum(r["score"] for r in results.values()) / len(results), 4)

    return JSONResponse({
        "baseline_agent": "greedy (highest-reward-first, deterministic)",
        "average_score":  avg,
        "tasks":          results,
    })

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

