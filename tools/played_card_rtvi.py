"""Declarative hosted-video tool for announcing played cards."""

TOOL_NAME = "played_card"
EXECUTION_MODE = "hosted_video_streaming"
VIDEO_CONFIG = {
    "window_seconds": 6,
    "interval_seconds": 3,
    "overlap_seconds": 3,
    "minimum_span_seconds": 5,
    "minimum_unique_frames": 12,
    "maximum_overlap_for_distinct_event": 0.45,
}
OUTPUT_CONFIG = {
    "schema": "played_card_event",
    "confidence_threshold": 0.8,
    "deduplicate": True,
    "cooldown_seconds": 5,
}
TOOL_PROMPT = """
When multiple chronological frames are available, establish the before hand
only from stable, sharp early frames and the after hand only from stable, sharp
final frames. Ignore the moving middle frames except for locating the action.
When only one image is available, use any clearly readable static card evidence
but do not infer that a card moved or disappeared. One image cannot prove a
played-card event, so return the uncertain no-event JSON described below.

A played card is valid only when the same specific rank and suit is clearly
visible in the hand before the action and clearly absent from the hand after the
action. Do not report the clearest remaining card. Never infer or invent a card
that was not readable before the action.

A card becoming newly exposed, moving to another position, rotating, or being
briefly covered is not a removal. Motion blur, fingers, glare, and temporary
occlusion are not evidence that a card disappeared. Both rank and suit of the
played card must be independently readable in stable early frames. Use only
ace, two, three, four, five, six, seven, eight, nine, ten, jack, queen, or king,
followed by "of clubs", "of diamonds", "of hearts", or "of spades". There are
no jokers. The evidence field must name only the played card and must agree
exactly with played_card.

If the same non-empty set of readable cards is visible before and after, report
event_detected false, played_card null, and high confidence. This means no card
was played. Do not use this result when the cards are unreadable.

Treat temporary occlusion, overlap, blur, rearrangement, camera motion, the hand
leaving the frame, or an unreadable corner as insufficient evidence. If the
before and after sets cannot both be determined reliably, set event_detected to
false, played_card to null, confidence no higher than 0.5, and do not claim that
the sets are unchanged.

Return only one JSON object with exactly these fields:
{
  "event_detected": true,
  "before_cards": ["jack of diamonds", "nine of hearts"],
  "after_cards": ["nine of hearts"],
  "played_card": "jack of diamonds",
  "confidence": 0.9,
  "evidence": "The jack of diamonds is readable before removal and absent afterward."
}

For a confidently unchanged hand, use the same JSON shape with event_detected
false, identical non-empty before_cards and after_cards, played_card null,
confidence at least 0.8, and evidence stating that the readable set did not
change. Do not use Markdown.
"""
