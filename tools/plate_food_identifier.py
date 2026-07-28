"""
Plate Food Identifier

Identifies food items on a plate and describes their positions using clock-face
directions for blind and low-vision users.

Audio output example:
  "Rice from 11 o'clock to 1 o'clock, mixed vegetables from 1 o'clock to 7
   o'clock, and sesame chicken from 7 o'clock to 11 o'clock."

Overlap handling: when two items share a clock-face region, they are ordered
clockwise; if they start at the same position, the one that ends soonest is
listed first.
"""

from __future__ import annotations

from model_router_client import copilot_llm_call

TOOL_NAME = "plate_food_identifier"


def main(image, input_data=None):
    if image is None:
        return "No image received. Please point the camera at your plate and try again."

    # Stage 1 – detect food items and their locations on the plate
    detection = copilot_llm_call(
        capability="object_detection_localization",
        goal=(
            "Detect every distinct food item visible on the plate in this image. "
            "For each item return its name and approximate bounding box or region "
            "so its position on the plate can be determined."
        ),
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "detect food items on a plate and locate each one",
            "target_labels": [
                "food", "rice", "chicken", "beef", "pork", "fish", "shrimp",
                "vegetables", "salad", "pasta", "bread", "soup", "sauce",
                "fruit", "dessert", "meat", "noodles", "egg", "cheese",
                "potato", "beans", "tofu", "sushi", "sandwich",
            ],
        },
    )

    detection_artifact = detection.get("artifact")

    # Stage 2 – map each detected food item to a clock-face range on the plate
    spatial = copilot_llm_call(
        capability="spatial_reasoning",
        goal=(
            "The image shows a plate of food viewed from above. "
            "Using the detected food items and their positions on the plate, "
            "describe where each food item is located using clock-face directions. "
            "Treat the plate like a clock: 12 o'clock is at the top of the plate, "
            "3 o'clock is at the right, 6 o'clock is at the bottom, and "
            "9 o'clock is at the left. All 12 positions (1 through 12) are valid "
            "because the plate is viewed from above, not as a camera navigation direction. "
            "For each food item state the clock-face arc it occupies, e.g. "
            "'Rice from 11 o'clock to 1 o'clock' or "
            "'mixed vegetables from 1 o'clock to 7 o'clock'. "
            "If two items share a clock-face region, list the one that starts "
            "earlier in clockwise order first; if they start at the same position, "
            "list the one that ends soonest first. "
            "Return a single, concise audio-friendly sentence suitable for "
            "text-to-speech, listing all items separated by commas and the last "
            "pair with 'and'. Do not include any JSON or bullet points."
        ),
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "map food items to clock-face positions on the plate",
            "previous_stage_artifact": detection_artifact,
        },
    )

    response = spatial.get("response", "").strip()
    if not response:
        return "I could not determine the food layout. Please try again with the plate clearly visible."

    return response
