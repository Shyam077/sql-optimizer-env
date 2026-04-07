from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class SQLAction(Action):
    optimization_type: str = Field(
        ...,
        description=(
            "Type of optimization to apply. One of: "
            "'reorder_joins' | 'add_index_hint' | 'push_predicate' | "
            "'eliminate_subquery' | 'remove_redundant_columns' | 'no_op'"
        ),
    )
    target_table: str = Field(default="", description="Table this optimization targets.")
    hint_value: str = Field(default="", description="Extra hint or index name.")


class SQLObservation(Observation):
    current_query: str = Field(..., description="Current SQL query after last action.")
    original_query: str = Field(..., description="Original unoptimized query.")
    schema_context: str = Field(..., description="Schema: tables, columns, indexes.")
    estimated_cost: float = Field(..., description="Simulated cost (lower is better).")
    original_cost: float = Field(..., description="Baseline cost for reward calculation.")
    is_valid_sql: bool = Field(..., description="Whether current query is valid SQL.")
    step_count: int = Field(..., description="Steps taken so far.")
    optimization_history: list = Field(default_factory=list, description="Optimizations applied.")
    done: bool = Field(default=False, description="Whether episode has ended.")