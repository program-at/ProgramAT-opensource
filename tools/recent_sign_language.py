"""Declarative temporal tool for recognizing a recent short signed gesture."""

TOOL_NAME = "recent_sign_language"
EXECUTION_MODE = "hosted_video_streaming"
VIDEO_CONFIG = {
    "window_seconds": 6,
    "interval_seconds": 3,
    "overlap_seconds": 3,
    "minimum_span_seconds": 4,
    "minimum_unique_frames": 8,
}
OUTPUT_CONFIG = {
    "deduplicate": True,
    "cooldown_seconds": 3,
}
TOOL_PROMPT = """
When multiple chronological recent frames are available, inspect them from
earliest to latest and track the signer's hand shapes, orientation, location,
facial cues, and motion sequence. When only one image is available, use only a
clearly recognizable held posture or other static evidence; do not infer motion
that is not visible. Identify the completed sign or short signed phrase and
return only one concise user-facing sentence, for example "The signer is waving
hello." If the available evidence is insufficient or no sign is clear, return
exactly "No clear sign detected." Do not return JSON, field names, analysis, or
card-related information.
"""
