import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uuid import uuid4
import random
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import SQLAction, SQLObservation

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


class SQLOptimizerEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario = None
        self._current_cost = 0.0
        self._original_cost = 0.0
        self._cost_factors_remaining = {}
        self._history = []
        self._current_query = ""

    def reset(self) -> SQLObservation:
        self._scenario = random.choice(QUERY_SCENARIOS)
        self._cost_factors_remaining = dict(self._scenario["cost_factors"])
        self._original_cost = sum(self._cost_factors_remaining.values()) + 10.0
        self._current_cost = self._original_cost
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._history = []
        self._current_query = self._scenario["original_query"]
        return self._make_observation(done=False)

    def step(self, action: SQLAction) -> SQLObservation:
        opt = action.optimization_type.strip().lower()
        reward = 0.0
        done = False

        if opt not in OPTIMIZATION_MAP:
            reward = -10.0
            done = True
            self._state.step_count += 1
            return self._make_observation(done=True)

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

        self._state.step_count += 1
        cost_ratio = self._current_cost / self._original_cost
        if self._state.step_count >= MAX_STEPS or cost_ratio <= (1 - COST_THRESHOLD_RATIO):
            done = True

        return self._make_observation(done=done)

    @property
    def state(self) -> State:
        return self._state

    def _make_observation(self, done: bool) -> SQLObservation:
        return SQLObservation(
            current_query=self._current_query,
            original_query=self._scenario["original_query"] if self._scenario else "",
            schema_context=self._scenario["schema"] if self._scenario else "",
            estimated_cost=round(self._current_cost, 2),
            original_cost=round(self._original_cost, 2),
            is_valid_sql=True,
            step_count=self._state.step_count,
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