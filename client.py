from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import SQLAction, SQLObservation


class SQLOptimizerEnv(EnvClient[SQLAction, SQLObservation, State]):
    """
    Client for the SQL Query Optimizer environment.

    Usage (async):
        async with SQLOptimizerEnv(base_url="https://your-space.hf.space") as env:
            obs = await env.reset()
            result = await env.step(SQLAction(optimization_type="eliminate_subquery"))

    Usage (sync):
        with SQLOptimizerEnv(base_url="https://your-space.hf.space").sync() as env:
            obs = env.reset()
            result = env.step(SQLAction(optimization_type="push_predicate"))
    """

    def _step_payload(self, action: SQLAction) -> Dict:
        return {
            "optimization_type": action.optimization_type,
            "target_table":      action.target_table,
            "hint_value":        action.hint_value,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SQLObservation]:
        obs_data = payload.get("observation", {})
        observation = SQLObservation(
            current_query        = obs_data.get("current_query", ""),
            original_query       = obs_data.get("original_query", ""),
            schema_context       = obs_data.get("schema_context", ""),
            estimated_cost       = obs_data.get("estimated_cost", 0.0),
            original_cost        = obs_data.get("original_cost", 0.0),
            is_valid_sql         = obs_data.get("is_valid_sql", True),
            step_count           = obs_data.get("step_count", 0),
            optimization_history = obs_data.get("optimization_history", []),
            done                 = payload.get("done", False),
        )
        return StepResult(
            observation = observation,
            reward      = payload.get("reward", 0.0),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id = payload.get("episode_id"),
            step_count = payload.get("step_count", 0),
        )
