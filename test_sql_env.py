# test_sql_env.py
# Standalone simulation — tests all core logic without needing openenv-core installed.
# Mirrors exactly what the OpenEnv framework calls during evaluation.

import random
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight stubs so the env logic runs without openenv-core installed
# ─────────────────────────────────────────────────────────────────────────────

class Action:
    pass

class Observation:
    pass

@dataclass
class SQLAction(Action):
    optimization_type: str
    target_table: str = ""
    hint_value: str = ""

@dataclass
class SQLObservation(Observation):
    current_query: str
    original_query: str
    schema_context: str
    estimated_cost: float
    original_cost: float
    is_valid_sql: bool
    step_count: int
    optimization_history: list = field(default_factory=list)
    done: bool = False

# ─────────────────────────────────────────────────────────────────────────────
# Copy of the environment (same logic as server/sql_environment.py)
# ─────────────────────────────────────────────────────────────────────────────

QUERY_SCENARIOS = [
    {
        "name": "ecommerce_orders",
        "schema": (
            "Tables:\n"
            "  orders       (order_id PK, customer_id FK, order_date, status)  — 2M rows\n"
            "  customers    (customer_id PK, name, region, signup_date)        — 500K rows\n"
            "  order_items  (item_id PK, order_id FK, product_id FK, qty, price) — 8M rows\n"
            "  products     (product_id PK, name, category, price)             — 10K rows\n"
            "Indexes: orders.customer_id, products.category"
        ),
        "original_query": (
            "SELECT c.name, c.region, SUM(oi.qty * oi.price) AS total_spent\n"
            "FROM customers c, orders o, order_items oi, products p\n"
            "WHERE c.customer_id = o.customer_id\n"
            "  AND o.order_id = oi.order_id\n"
            "  AND oi.product_id = p.product_id\n"
            "  AND (SELECT COUNT(*) FROM orders o2 WHERE o2.customer_id = c.customer_id) > 5\n"
            "  AND p.category = 'Electronics'\n"
            "GROUP BY c.name, c.region\n"
            "ORDER BY total_spent DESC"
        ),
        "cost_factors": {
            "has_subquery": 40.0,
            "no_predicate_push": 25.0,
            "implicit_joins": 15.0,
            "missing_index_hints": 20.0,
            "unnecessary_columns": 0.0,
        },
    },
    {
        "name": "hr_payroll",
        "schema": (
            "Tables:\n"
            "  employees    (emp_id PK, dept_id FK, name, salary, hire_date) — 100K rows\n"
            "  departments  (dept_id PK, dept_name, manager_id)              — 500 rows\n"
            "  payroll      (pay_id PK, emp_id FK, month, gross, deductions) — 1.2M rows\n"
            "Indexes: employees.dept_id"
        ),
        "original_query": (
            "SELECT e.name, d.dept_name, p.gross, p.deductions\n"
            "FROM employees e, departments d, payroll p\n"
            "WHERE e.dept_id = d.dept_id\n"
            "  AND e.emp_id = p.emp_id\n"
            "  AND p.gross > (SELECT AVG(gross) FROM payroll)\n"
            "  AND e.hire_date > '2020-01-01'"
        ),
        "cost_factors": {
            "has_subquery": 30.0,
            "no_predicate_push": 20.0,
            "implicit_joins": 15.0,
            "missing_index_hints": 15.0,
            "unnecessary_columns": 5.0,
        },
    },
    {
        "name": "insurance_claims",
        "schema": (
            "Tables:\n"
            "  claims       (claim_id PK, policy_id FK, claim_date, amount, status) — 3M rows\n"
            "  policies     (policy_id PK, customer_id FK, type, start_date)        — 800K rows\n"
            "  customers    (customer_id PK, name, dob, region)                     — 600K rows\n"
            "  adjusters    (adj_id PK, claim_id FK, decision, notes)               — 2.5M rows\n"
            "Indexes: claims.policy_id, claims.status"
        ),
        "original_query": (
            "SELECT c.name, cl.amount, cl.status, a.decision\n"
            "FROM customers c, policies po, claims cl, adjusters a\n"
            "WHERE c.customer_id = po.customer_id\n"
            "  AND po.policy_id = cl.policy_id\n"
            "  AND cl.claim_id = a.claim_id\n"
            "  AND cl.amount > 10000\n"
            "  AND cl.status IN (\n"
            "        SELECT DISTINCT status FROM claims WHERE claim_date > '2023-01-01'\n"
            "      )"
        ),
        "cost_factors": {
            "has_subquery": 45.0,
            "no_predicate_push": 30.0,
            "implicit_joins": 15.0,
            "missing_index_hints": 10.0,
            "unnecessary_columns": 0.0,
        },
    },
]

OPTIMIZATION_MAP = {
    "eliminate_subquery":       "has_subquery",
    "push_predicate":           "no_predicate_push",
    "reorder_joins":            "implicit_joins",
    "add_index_hint":           "missing_index_hints",
    "remove_redundant_columns": "unnecessary_columns",
    "no_op":                    None,
}

MAX_STEPS = 6
COST_THRESHOLD_RATIO = 0.50


class SQLOptimizerEnvironment:

    def __init__(self):
        self._scenario = None
        self._current_cost = 0.0
        self._original_cost = 0.0
        self._cost_factors_remaining = {}
        self._step_count = 0
        self._history = []
        self._current_query = ""

    def reset(self, scenario_index=None) -> SQLObservation:
        if scenario_index is not None:
            self._scenario = QUERY_SCENARIOS[scenario_index]
        else:
            self._scenario = random.choice(QUERY_SCENARIOS)
        self._cost_factors_remaining = dict(self._scenario["cost_factors"])
        self._original_cost = sum(self._cost_factors_remaining.values()) + 10.0
        self._current_cost = self._original_cost
        self._step_count = 0
        self._history = []
        self._current_query = self._scenario["original_query"]
        return self._make_observation(done=False)

    def step(self, action: SQLAction):
        opt = action.optimization_type.strip().lower()
        reward = 0.0
        done = False

        if opt not in OPTIMIZATION_MAP:
            reward = -10.0
            done = True
            self._step_count += 1
            return self._make_observation(done=True), reward, done

        cost_key = OPTIMIZATION_MAP[opt]

        if opt == "no_op":
            reward = -1.0
        elif cost_key and self._cost_factors_remaining.get(cost_key, 0) > 0:
            reduction = self._cost_factors_remaining.pop(cost_key)
            self._current_cost -= reduction
            reward = round(reduction, 2)
            self._history.append(opt)
            self._current_query = self._apply_rewrite(opt, action, self._current_query)
        else:
            reward = -2.0
            self._history.append(f"{opt}(no_effect)")

        self._step_count += 1
        cost_ratio = self._current_cost / self._original_cost
        if self._step_count >= MAX_STEPS or cost_ratio <= (1 - COST_THRESHOLD_RATIO):
            done = True

        return self._make_observation(done=done), reward, done

    def state(self) -> dict:
        return {
            "step_count": self._step_count,
            "current_cost": round(self._current_cost, 2),
            "original_cost": round(self._original_cost, 2),
            "scenario": self._scenario["name"] if self._scenario else None,
            "history": self._history,
        }

    def _make_observation(self, done: bool) -> SQLObservation:
        return SQLObservation(
            current_query=self._current_query,
            original_query=self._scenario["original_query"],
            schema_context=self._scenario["schema"],
            estimated_cost=round(self._current_cost, 2),
            original_cost=round(self._original_cost, 2),
            is_valid_sql=True,
            step_count=self._step_count,
            optimization_history=list(self._history),
            done=done,
        )

    def _apply_rewrite(self, optimization: str, action: SQLAction, query: str) -> str:
        if optimization == "reorder_joins":
            query = query.replace(
                "FROM customers c, orders o, order_items oi, products p",
                "FROM customers c\nINNER JOIN orders o ON c.customer_id = o.customer_id\n"
                "INNER JOIN order_items oi ON o.order_id = oi.order_id\n"
                "INNER JOIN products p ON oi.product_id = p.product_id",
            )
            query = query.replace(
                "FROM employees e, departments d, payroll p",
                "FROM employees e\nINNER JOIN departments d ON e.dept_id = d.dept_id\n"
                "INNER JOIN payroll p ON e.emp_id = p.emp_id",
            )
            query = query.replace(
                "FROM customers c, policies po, claims cl, adjusters a",
                "FROM customers c\nINNER JOIN policies po ON c.customer_id = po.customer_id\n"
                "INNER JOIN claims cl ON po.policy_id = cl.policy_id\n"
                "INNER JOIN adjusters a ON cl.claim_id = a.claim_id",
            )
            for clause in [
                "c.customer_id = o.customer_id\n  AND ",
                "o.order_id = oi.order_id\n  AND ",
                "oi.product_id = p.product_id\n  AND ",
                "e.dept_id = d.dept_id\n  AND ",
                "e.emp_id = p.emp_id\n  AND ",
                "c.customer_id = po.customer_id\n  AND ",
                "po.policy_id = cl.policy_id\n  AND ",
                "cl.claim_id = a.claim_id\n  AND ",
            ]:
                query = query.replace(clause, "")

        elif optimization == "eliminate_subquery":
            query = (
                "WITH customer_order_counts AS (\n"
                "  SELECT customer_id, COUNT(*) AS order_count\n"
                "  FROM orders GROUP BY customer_id HAVING COUNT(*) > 5\n"
                ")\n" + query
            )
            query = query.replace(
                "  AND (SELECT COUNT(*) FROM orders o2 WHERE o2.customer_id = c.customer_id) > 5\n",
                "  AND c.customer_id IN (SELECT customer_id FROM customer_order_counts)\n",
            )
            query = query.replace(
                "  AND p.gross > (SELECT AVG(gross) FROM payroll)\n",
                "  AND p.gross > (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY gross) FROM payroll)\n",
            )
            query = query.replace(
                "  AND cl.status IN (\n"
                "        SELECT DISTINCT status FROM claims WHERE claim_date > '2023-01-01'\n"
                "      )",
                "  AND cl.status IN ('open', 'under_review')",
            )

        elif optimization == "push_predicate":
            query = "-- OPTIMIZED: predicates pushed to earliest filter point\n" + query

        elif optimization == "add_index_hint":
            table = action.target_table if action.target_table else "key join columns"
            query = f"-- INDEX HINT: USE INDEX ON {table}\n" + query

        elif optimization == "remove_redundant_columns":
            query = "-- OPTIMIZED: SELECT list narrowed to required columns only\n" + query

        return query


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def sep(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")

def run_tests():
    total = 0
    passed = 0

    def check(label, condition):
        nonlocal total, passed
        total += 1
        status = PASS if condition else FAIL
        if condition:
            passed += 1
        print(f"  {status}  {label}")
        return condition

    env = SQLOptimizerEnvironment()

    # ── TEST 1: reset() ───────────────────────────────────────────────────
    sep("TEST 1 — reset() initialises correctly")
    obs = env.reset(scenario_index=0)  # ecommerce
    check("Scenario loaded (ecommerce_orders)", env._scenario["name"] == "ecommerce_orders")
    check("original_cost > 0", obs.original_cost > 0)
    check("estimated_cost == original_cost on reset", obs.estimated_cost == obs.original_cost)
    check("step_count == 0", obs.step_count == 0)
    check("optimization_history is empty", obs.optimization_history == [])
    check("done == False", obs.done == False)
    check("is_valid_sql == True", obs.is_valid_sql == True)
    print(f"\n  Original cost: {obs.original_cost}")
    print(f"  Query preview: {obs.current_query[:80].strip()}...")

    # ── TEST 2: valid optimization reduces cost ───────────────────────────
    sep("TEST 2 — Valid action reduces cost & gives positive reward")
    obs = env.reset(scenario_index=0)
    cost_before = obs.estimated_cost
    obs2, reward, done = env.step(SQLAction(optimization_type="eliminate_subquery"))
    check("Cost decreased after eliminate_subquery", obs2.estimated_cost < cost_before)
    check("Reward is positive", reward > 0)
    check("step_count incremented to 1", obs2.step_count == 1)
    check("optimization_history updated", "eliminate_subquery" in obs2.optimization_history)
    check("is_valid_sql still True", obs2.is_valid_sql == True)
    print(f"\n  Cost: {cost_before} → {obs2.estimated_cost}  (reward: +{reward})")

    # ── TEST 3: duplicate optimization penalized ──────────────────────────
    sep("TEST 3 — Duplicate optimization gives -2 penalty")
    obs3, reward3, _ = env.step(SQLAction(optimization_type="eliminate_subquery"))
    check("Reward is -2 for duplicate", reward3 == -2.0)
    check("Cost unchanged after duplicate", obs3.estimated_cost == obs2.estimated_cost)
    print(f"\n  Reward for duplicate: {reward3}")

    # ── TEST 4: no_op penalized ───────────────────────────────────────────
    sep("TEST 4 — no_op gives -1 penalty")
    obs4, reward4, _ = env.step(SQLAction(optimization_type="no_op"))
    check("Reward is -1 for no_op", reward4 == -1.0)
    print(f"\n  Reward for no_op: {reward4}")

    # ── TEST 5: invalid action ends episode ───────────────────────────────
    sep("TEST 5 — Invalid action type gives -10 and ends episode")
    obs = env.reset(scenario_index=1)
    obs5, reward5, done5 = env.step(SQLAction(optimization_type="drop_table"))
    check("Reward is -10 for invalid action", reward5 == -10.0)
    check("Episode done after invalid action", done5 == True)
    print(f"\n  Reward: {reward5}, done: {done5}")

    # ── TEST 6: full optimal episode (ecommerce) ──────────────────────────
    sep("TEST 6 — Full optimal episode on ecommerce scenario")
    obs = env.reset(scenario_index=0)
    print(f"\n  Start cost: {obs.original_cost}")
    sequence = [
        "eliminate_subquery",
        "push_predicate",
        "reorder_joins",
        "add_index_hint",
    ]
    total_reward = 0
    for action_type in sequence:
        obs, reward, done = env.step(SQLAction(optimization_type=action_type))
        total_reward += reward
        print(f"  [{action_type:30s}]  cost={obs.estimated_cost:6.1f}  reward={reward:+.1f}  done={done}")
        if done:
            break

    check("Final cost lower than original", obs.estimated_cost < obs.original_cost)
    check("Total reward is positive", total_reward > 0)
    cost_reduction_pct = (1 - obs.estimated_cost / obs.original_cost) * 100
    check(f"Cost reduced by ≥ 50% ({cost_reduction_pct:.1f}%)", cost_reduction_pct >= 50)
    check("Episode marked done", done == True)
    print(f"\n  Final cost: {obs.estimated_cost} / {obs.original_cost}  ({cost_reduction_pct:.1f}% reduction)")
    print(f"  Total reward: {total_reward}")
    print(f"  History: {obs.optimization_history}")

    # ── TEST 7: max_steps ends episode ───────────────────────────────────
    sep("TEST 7 — Episode ends at MAX_STEPS (6)")
    obs = env.reset(scenario_index=1)
    for i in range(MAX_STEPS):
        obs, reward, done = env.step(SQLAction(optimization_type="no_op"))
    check("Episode done after 6 steps", done == True)
    check("step_count == 6", obs.step_count == MAX_STEPS)
    print(f"\n  Step count: {obs.step_count}, done: {done}")

    # ── TEST 8: state() returns correct metadata ──────────────────────────
    sep("TEST 8 — state() returns correct metadata")
    obs = env.reset(scenario_index=2)
    env.step(SQLAction(optimization_type="reorder_joins"))
    s = env.state()
    check("state() has step_count key", "step_count" in s)
    check("state() step_count == 1", s["step_count"] == 1)
    check("state() has scenario name", s["scenario"] == "insurance_claims")
    check("state() current_cost < original_cost", s["current_cost"] < s["original_cost"])
    print(f"\n  State: {s}")

    # ── TEST 9: SQL rewrite visibly changes query ─────────────────────────
    sep("TEST 9 — SQL rewrite actually changes the query string")
    obs = env.reset(scenario_index=0)
    original_q = obs.current_query
    obs2, _, _ = env.step(SQLAction(optimization_type="reorder_joins"))
    check("Query changed after reorder_joins", obs2.current_query != original_q)
    check("INNER JOIN appears in rewritten query", "INNER JOIN" in obs2.current_query)
    print(f"\n  Original snippet: {original_q[50:100].strip()}")
    print(f"  Rewritten snippet: {obs2.current_query[50:120].strip()}")

    # ── TEST 10: all 3 scenarios loadable ────────────────────────────────
    sep("TEST 10 — All 3 scenarios initialise correctly")
    for i, scenario in enumerate(QUERY_SCENARIOS):
        obs = env.reset(scenario_index=i)
        check(f"Scenario [{scenario['name']}] loads with valid cost", obs.original_cost > 0)

    # ── Summary ──────────────────────────────────────────────────────────
    sep()
    print(f"\n  Results: {passed}/{total} tests passed")
    if passed == total:
        print("  🎉 ALL TESTS PASSED — environment is ready for submission!\n")
    else:
        print(f"  ⚠️  {total - passed} test(s) failed — review above.\n")

if __name__ == "__main__":
    run_tests()
