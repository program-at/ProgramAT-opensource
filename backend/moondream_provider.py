"""Shared, streaming-independent Moondream Cloud request helpers."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping

BASE_URL = "https://api.moondream.ai/v1"
MESSAGE_FORMAT = "nested_image_url_url"
SHORT_GOAL_MAX_CHARS = 160
PROMPT_MAX_CHARS = 250
CARD_IDENTIFICATION_PROMPT = (
    "Read only the top-left index of each physical card. Ignore all other corners. "
    "Rank+suit only. Multiple cards: left to right."
)
CARD_PROMPT_VARIANT = "multi_card_left_to_right"
CARD_FALSE_POSITIVE_TERMS = ("credit card", "id card", "business card", "gift card")
CARD_RANK_PATTERN = r"(?:ace|king|queen|jack|ten|nine|eight|seven|six|five|four|three|two|10|[2-9]|[akqj])"
CARD_SUIT_PATTERN = r"(?:spades?|hearts?|diamonds?|clubs?)"
CARD_PAIR_RE = re.compile(
    rf"\b(?P<rank>{CARD_RANK_PATTERN})\s+of\s+(?P<suit>{CARD_SUIT_PATTERN})\b",
    re.IGNORECASE,
)
COMPACT_CARD_SYMBOL_RE = re.compile(
    r"(?<!\w)(?P<rank>10|[2-9AKQJ])\s*(?P<suit>[♠♥♦♣])(?!\w)",
    re.IGNORECASE,
)
COMPACT_CARD_LETTER_RE = re.compile(
    r"(?<!\w)(?P<rank>10|[2-9AKQJ])\s*(?:-\s*)?(?P<suit>[SHDC])(?!\w)",
)
COMPACT_CARD_WORD_RE = re.compile(
    r"(?<!\w)(?P<rank>10|[2-9AKQJ])(?:\s*-\s*|\s+)"
    r"(?P<suit>spades?|hearts?|diamonds?|clubs?)(?!\w)",
    re.IGNORECASE,
)
CARD_RANK_WORD_TOKEN_RE = re.compile(
    r"\b(?:ace|king|queen|jack|ten|nine|eight|seven|six|five|four|three|two|10|[2-9])\b",
    re.IGNORECASE,
)
CARD_RANK_LETTER_TOKEN_RE = re.compile(r"(?<!\w)[AKQJ](?!\w)")
CARD_SUIT_TOKEN_RE = re.compile(
    r"\b(?:spades?|hearts?|diamonds?|clubs?)\b|[♠♥♦♣]",
    re.IGNORECASE,
)
CARD_RANK_LABELS = {
    "a": "Ace", "ace": "Ace",
    "k": "King", "king": "King",
    "q": "Queen", "queen": "Queen",
    "j": "Jack", "jack": "Jack",
    "two": "Two", "three": "Three", "four": "Four", "five": "Five",
    "six": "Six", "seven": "Seven", "eight": "Eight", "nine": "Nine",
    "ten": "Ten",
    "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
    "7": "Seven", "8": "Eight", "9": "Nine", "10": "Ten",
}
CARD_SUIT_LABELS = {
    "s": "spades", "spade": "spades", "spades": "spades", "♠": "spades",
    "h": "hearts", "heart": "hearts", "hearts": "hearts", "♥": "hearts",
    "d": "diamonds", "diamond": "diamonds", "diamonds": "diamonds", "♦": "diamonds",
    "c": "clubs", "club": "clubs", "clubs": "clubs", "♣": "clubs",
}
MIRRORED_CORNER_TERMS = (
    "lower-right", "lower right", "lower corner", "bottom-right", "bottom right",
    "upside-down", "upside down", "mirrored corner", "corner index", "corner indexes",
)


def input_type(image: Any) -> str:
    if isinstance(image, str) and image.startswith("data:image"):
        return "data_uri"
    if isinstance(image, (bytes, bytearray)):
        return "bytes"
    from PIL import Image
    import numpy as np
    if isinstance(image, np.ndarray):
        return "ndarray"
    if isinstance(image, Image.Image):
        return "PIL"
    return type(image).__name__


def image_bytes(image: Any) -> bytes:
    output = io.BytesIO()
    to_rgb_image(image).save(output, format="JPEG")
    return output.getvalue()


def to_rgb_image(image: Any):
    from PIL import Image, ImageOps

    if isinstance(image, str):
        payload = image.split(",", 1)[1] if image.startswith("data:image") and "," in image else image
        return ImageOps.exif_transpose(
            Image.open(io.BytesIO(base64.b64decode(payload)))
        ).convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return ImageOps.exif_transpose(Image.open(io.BytesIO(bytes(image)))).convert("RGB")
    import numpy as np
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            pil_image = Image.fromarray(array, mode="L").convert("RGB")
        elif array.ndim == 3 and array.shape[2] == 1:
            pil_image = Image.fromarray(array[:, :, 0], mode="L").convert("RGB")
        elif array.ndim == 3 and array.shape[2] == 3:
            # stream_server.decode_frame() returns OpenCV BGR arrays.
            pil_image = Image.fromarray(array[:, :, ::-1].copy(), mode="RGB")
        elif array.ndim == 3 and array.shape[2] == 4:
            # Preserve alpha long enough for Pillow to composite via RGB conversion.
            pil_image = Image.fromarray(array[:, :, [2, 1, 0, 3]].copy(), mode="RGBA").convert("RGB")
        else:
            raise TypeError(f"Unsupported ndarray image shape: {array.shape}")
        return pil_image.convert("RGB")
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def prepare_image(image: Any) -> tuple[bytes, str, tuple[int, int]]:
    max_dimension = max(1, int(os.environ.get("MOONDREAM_MAX_IMAGE_DIMENSION", "768")))
    quality = min(95, max(1, int(os.environ.get("MOONDREAM_JPEG_QUALITY", "75"))))
    return prepare_rgb_jpeg(image, max_dimension, quality)


def prepare_rgb_jpeg(
    image: Any, max_dimension: int, quality: int
) -> tuple[bytes, str, tuple[int, int]]:
    from PIL import Image, ImageOps

    normalized = ImageOps.exif_transpose(to_rgb_image(image)).convert("RGB")
    normalized.thumbnail((max(1, max_dimension), max(1, max_dimension)), Image.Resampling.LANCZOS)
    dimensions = normalized.size
    output = io.BytesIO()
    normalized.save(
        output, format="JPEG", quality=min(95, max(1, quality)), optimize=True
    )
    return output.getvalue(), "image/jpeg", dimensions


def image_metrics(image: Any) -> tuple[tuple[int, int], int]:
    import numpy as np

    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        return (width, height), int(image.nbytes)
    if isinstance(image, (bytes, bytearray)):
        return to_rgb_image(image).size, len(image)
    if isinstance(image, str):
        payload = image.split(",", 1)[1] if "," in image else image
        raw = base64.b64decode(payload)
        return to_rgb_image(raw).size, len(raw)
    converted = to_rgb_image(image)
    return converted.size, len(image_bytes(converted))


def build_messages(
    messages: Iterable[Mapping[str, Any]], image_data_uri: str
) -> List[Dict[str, Any]]:
    request_messages = [dict(message) for message in messages]
    image_part = {"type": "image_url", "image_url": {"url": image_data_uri}}
    user_index = next(
        (index for index in range(len(request_messages) - 1, -1, -1)
         if request_messages[index].get("role") == "user"), None,
    )
    if user_index is None:
        request_messages.append({"role": "user", "content": [image_part]})
    else:
        content = request_messages[user_index].get("content", "")
        text_parts = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        text_parts = [
            part for part in text_parts
            if not isinstance(part, dict) or part.get("type") != "image_url"
        ]
        # Documented production default: text first, then nested image_url.url.
        request_messages[user_index]["content"] = [*text_parts, image_part]
    return request_messages


def prompt_text(messages: Iterable[Mapping[str, Any]]) -> str:
    for message in reversed(list(messages)):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text") or "").strip()
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
    return ""


def all_prompt_text(messages: Iterable[Mapping[str, Any]]) -> str:
    parts = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return "\n".join(part for part in parts if part)


def normalize_short_goal(value: Any, max_chars: int = SHORT_GOAL_MAX_CHARS) -> str:
    text = str(value or "")
    text = re.sub(r"```(?:\w+)?", " ", text)
    text = text.replace("`", " ").replace("**", "").replace("__", "")
    text = re.sub(r"(?m)^\s*(?:#{1,6}|[-*+] |\d+[.)] )", "", text)
    text = " ".join(text.split()).strip(" -:;,")
    boilerplate_markers = (
        "return only concise", "do not use markdown", "output exactly", "output only",
        "execution policy", "routing detail", "evaluator", "json schema",
        "stage handoff", "previous-stage artifact", "live camera stream",
        "audio-friendly plain text", "blind or low-vision user",
        "for ordered information",
    )
    clauses = re.split(r"(?<=[.!?])\s+", text)
    useful = [
        clause.strip()
        for clause in clauses
        if clause.strip() and not any(marker in clause.lower() for marker in boilerplate_markers)
    ]
    task = useful[0] if useful else ""
    if len(task) <= max_chars:
        return task
    window = task[: max_chars + 1]
    boundary = max(window.rfind(mark) for mark in (";", ",", ":"))
    if boundary < max_chars // 2:
        boundary = window.rfind(" ", 0, max_chars + 1)
    if boundary <= 0:
        boundary = max_chars
    shortened = window[:boundary].rstrip(" ,;:-")
    if shortened and shortened[-1] not in ".!?":
        shortened += "."
    return shortened


def card_mode_reason(value: Any) -> tuple[bool, str]:
    text = " ".join(str(value or "").lower().split())
    if not text:
        return False, "empty task"
    for false_positive in CARD_FALSE_POSITIVE_TERMS:
        if false_positive in text:
            return False, f"excluded phrase: {false_positive}"
    card_terms = ("playing card", "playing cards", "card game", "poker")
    for term in card_terms:
        if term in text:
            return True, f"matched phrase: {term}"
    if "rank" in text or "suit" in text:
        return True, "matched rank/suit"
    if "identify" in text and "cards" in text:
        return True, "matched identify+cards"
    if "card identifier" in text:
        return True, "matched card identifier"
    if "cards" in text and ("hand" in text or "visible" in text):
        return True, "matched cards+hand/visible"
    return False, "no playing-card trigger"


def is_card_identification_task(value: Any) -> bool:
    return card_mode_reason(value)[0]


def card_response_details(value: Any) -> Dict[str, Any]:
    text = str(value or "")
    matches = []
    for match in CARD_PAIR_RE.finditer(text):
        matches.append((match.start(), match.group("rank"), match.group("suit"), False))
    for pattern in (COMPACT_CARD_SYMBOL_RE, COMPACT_CARD_LETTER_RE, COMPACT_CARD_WORD_RE):
        for match in pattern.finditer(text):
            matches.append((match.start(), match.group("rank"), match.group("suit"), True))
    matches.sort(key=lambda item: item[0])

    extracted_cards = []
    for _start, rank, suit, _compact in matches:
        rank_label = CARD_RANK_LABELS.get(rank.lower(), rank.capitalize())
        suit_label = CARD_SUIT_LABELS.get(suit.lower(), suit.lower().rstrip("s") + "s")
        extracted_cards.append(f"{rank_label} of {suit_label}")

    deduped_cards = []
    seen = set()
    for card in extracted_cards:
        key = card.lower()
        if key not in seen:
            deduped_cards.append(card)
            seen.add(key)

    lowered = text.lower()
    mirrored_corner_filter_applied = False
    final_cards = deduped_cards
    if len(deduped_cards) == 2:
        has_corner_cue = any(term in lowered for term in MIRRORED_CORNER_TERMS)
        first_match_start = matches[0][0] if matches else None
        singular_card_intro = bool(
            first_match_start is not None
            and re.search(r"\bthe card\b", lowered[:first_match_start])
        )
        if has_corner_cue or singular_card_intro:
            final_cards = deduped_cards[:1]
            mirrored_corner_filter_applied = True

    final_response = ", ".join(final_cards) + "." if final_cards else "Uncertain."
    rank_tokens = [match.group(0) for match in CARD_RANK_WORD_TOKEN_RE.finditer(text)]
    rank_tokens.extend(match.group(0) for match in CARD_RANK_LETTER_TOKEN_RE.finditer(text))
    suit_tokens = [match.group(0) for match in CARD_SUIT_TOKEN_RE.finditer(text)]
    suit_tokens.extend(
        match.group("suit") for match in COMPACT_CARD_LETTER_RE.finditer(text)
    )
    conflicting_suits = suit_tokens if len(suit_tokens) > len(matches) else []
    discarded_card_tokens = []
    if len(rank_tokens) > len(matches):
        discarded_card_tokens.extend(rank_tokens)
    if conflicting_suits:
        discarded_card_tokens.extend(conflicting_suits)
    explicit_uncertainty = bool(re.search(r"\buncertain\b", text, re.IGNORECASE))
    card_result_valid = bool(
        final_cards and not explicit_uncertainty and not discarded_card_tokens
    )
    card_result_clean = bool(
        card_result_valid
        and not discarded_card_tokens
        and len(rank_tokens) == len(matches)
        and len(suit_tokens) == len(matches)
    )
    return {
        "raw_response": text,
        "extracted_cards": extracted_cards,
        "deduped_cards": deduped_cards,
        "discarded_card_tokens": discarded_card_tokens,
        "conflicting_suits": conflicting_suits,
        "compact_notation_detected": any(match[3] for match in matches),
        "mirrored_corner_filter_applied": mirrored_corner_filter_applied,
        "card_result_valid": card_result_valid,
        "card_result_clean": card_result_clean,
        "final_response": final_response,
    }


def normalize_card_response(value: Any) -> str:
    return card_response_details(value)["final_response"]


def adapt_task_prompt(
    messages: Iterable[Mapping[str, Any]], metadata: Mapping[str, Any] | None = None
) -> tuple[str, str, int, str]:
    """Return the extractive prompt, short task, source length, and source label."""
    message_list = list(messages)
    metadata = metadata or {}
    stage_goal = next(
        (metadata.get(key) for key in ("stage_goal", "step_goal") if metadata.get(key)),
        None,
    )
    user_goal = next(
        (
            metadata.get(key)
            for key in ("goal", "user_goal", "tool_goal", "task_text", "task")
            if metadata.get(key)
        ),
        None,
    )
    if stage_goal:
        raw_task, source = str(stage_goal), "stage_goal"
    elif user_goal:
        raw_task, source = str(user_goal), "user_goal"
    else:
        raw_task = prompt_text(message_list) or all_prompt_text(message_list)
        source = "original_prompt"
    if is_card_identification_task(raw_task):
        short_task = normalize_short_goal(raw_task)
        return CARD_IDENTIFICATION_PROMPT, short_task, len(raw_task), source
    capability = str(metadata.get("capability") or "").strip().lower()
    prefix = (
        "Read visible text needed for this task and answer briefly:"
        if capability == "ocr"
        else "Answer the visual task directly and briefly:"
    )
    available = PROMPT_MAX_CHARS - len(prefix) - 1
    short_task = normalize_short_goal(raw_task, max_chars=available)
    if not short_task:
        short_task = "Describe the relevant visual information."
    adapted = f"{prefix} {short_task}"
    return adapted[:PROMPT_MAX_CHARS], short_task, len(raw_task), source


def create_client(api_key: str, timeout: float):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=timeout, max_retries=0)


def response_text(completion: Any) -> str:
    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            str(item.get("text") or "").strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    return ""


def error_context(exc: Exception) -> tuple[Any, str, str]:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if request_id is None and response is not None:
        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
    body = getattr(response, "text", None) or getattr(exc, "body", None) or str(exc)
    if not isinstance(body, str):
        body = json.dumps(body, ensure_ascii=False, default=str)
    return status, request_id or "unavailable", body
