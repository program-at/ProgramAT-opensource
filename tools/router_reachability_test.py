"""Minimal tool for verifying access to the backend model router through the client."""

from __future__ import annotations

from model_router_client import llm_call


TOOL_NAME = "router_reachability_test"


def main(image, input_data=None):
    prompt = "Reply with one short sentence confirming that the image was received."
    return llm_call(
        task_category="visual_understanding",
        messages=[
            {"role": "system", "content": "Keep responses concise and audio-friendly."},
            {"role": "user", "content": prompt},
        ],
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "minimal generated tool reachability test for a visual understanding call",
        },
    )
