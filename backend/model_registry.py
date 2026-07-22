"""Small, inspectable registry of models available to generated tool policies."""

MODEL_REGISTRY = {
    "yolo": {
        "provider_model": "yolo11n.pt",
        "executor": "yolo",
        "supported_capabilities": ["object_detection", "object_localization"],
        "relative_latency": "very_low",
        "relative_accuracy": "medium",
        "cost": "local",
        "limitations": "Object detection only; limited to known detector classes.",
        "recommended_use_cases": ["Fast local object presence and location checks."],
    },
    "moondream": {
        "provider_model": "moondream/moondream3-preview",
        "executor": "moondream_cloud",
        "supported_capabilities": ["simple_visual_qa", "image_description"],
        "relative_latency": "low",
        "relative_accuracy": "low_to_medium",
        "cost": "low",
        "limitations": "Not recommended when fine detail or high confidence is required.",
        "recommended_use_cases": ["Simple visual tasks with low accuracy requirements."],
    },
    "gemini-3.1-flash-lite": {
        "provider_model": "gemini/gemini-3.1-flash-lite-preview",
        "executor": "model",
        "supported_capabilities": ["visual_qa", "ocr", "reasoning", "image_description"],
        "relative_latency": "low",
        "relative_accuracy": "high",
        "cost": "low",
        "limitations": "Less reliable than GPT-5 on the hardest visual reasoning tasks.",
        "recommended_use_cases": ["Default general-purpose vision-language tasks."],
    },
    "gpt-5": {
        "provider_model": "gpt-5",
        "executor": "openai_responses",
        "supported_capabilities": ["visual_qa", "ocr", "complex_reasoning", "image_description"],
        "relative_latency": "high",
        "relative_accuracy": "very_high",
        "cost": "high",
        "limitations": "High latency and cost.",
        "recommended_use_cases": ["Tasks requiring the strongest reasoning or visual accuracy."],
    },
    "gpt-4o-mini": {
        "provider_model": "openai/gpt-4o-mini",
        "executor": "model",
        "supported_capabilities": ["text_evaluation", "result_aggregation", "lightweight_reasoning"],
        "relative_latency": "low",
        "relative_accuracy": "medium",
        "cost": "low",
        "limitations": "Not the default vision model.",
        "recommended_use_cases": ["Evaluating or aggregating candidate results."],
    },
}

DEFAULT_TAKE_PHOTO_POLICY = {
    "strategy": "single",
    "models": ["gemini-3.1-flash-lite"],
}

# Infrastructure calls are deliberately separate from generated visual tool policies.
SYSTEM_MODEL = "groq/llama-3.1-8b-instant"

