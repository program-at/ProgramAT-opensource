"""Schemas for execution strategies accepted by the shared policy executor."""

STRATEGY_REGISTRY = {
    "single": {
        "description": "Run exactly one model once.",
        "required": ["models"],
        "schema": {"models": "list[str] (exactly one)"},
    },
    "cascade": {
        "description": "Run models in order until a result passes evaluation.",
        "required": ["models", "evaluator"],
        "schema": {
            "models": "ordered list[str]",
            "evaluator": "model name",
            "stop_condition": "accepted | non_empty",
        },
    },
    "parallel_first": {
        "description": "Run models concurrently and return the first acceptable result.",
        "required": ["models"],
        "schema": {
            "models": "list[str]",
            "evaluator": "model name (optional)",
            "stop_condition": "first_complete | accepted (optional)",
        },
    },
    "parallel_aggregate": {
        "description": "Run models concurrently and aggregate their results.",
        "required": ["models", "aggregator"],
        "schema": {
            "models": "list[str]",
            "aggregator": "model name",
            "aggregation_prompt": "string (optional)",
        },
    },
    "parallel_progressive": {
        "description": "Run models concurrently and emit every successful result in completion order.",
        "required": ["models"],
        "schema": {
            "models": "list[str]",
        },
    },
    "conditional": {
        "description": "Select a nested policy from an explicit runtime condition.",
        "required": ["condition", "if_true", "if_false"],
        "schema": {
            "models": "list[str] (optional models used by nested policies)",
            "condition": "metadata key or {key, equals}",
            "if_true": "tool policy",
            "if_false": "tool policy",
        },
    },
}
