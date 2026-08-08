"""
Appearance Check Tool ("Do I Look OK?")

AI-powered appearance check for blind or low vision users. The person points the
phone's camera at themselves and hears an honest, verifiable answer to a single
question: "Is anything wrong with how I look before I go out?"

Features:
- Checks objective, fixable problems: clothing inside-out or backwards, uneven
  buttons, an open zipper, stains, a tag sticking out, a folded collar, or lint
  and other debris stuck on the clothing.
- Checks framing first. If the photo is too dark or badly framed, it explains
  how to retake it instead of guessing.
- Detect-then-verify pipeline: a second, focused look re-checks every finding to
  cut false alarms, which are what make blind users distrust this kind of tool.
- Gives each issue a body-relative location the person can touch to confirm it.
- Never gives false reassurance. On any error it asks for a retry; it only says
  "you look good" when it actually completed the check.

Audio Output:
- Returns natural language suitable for text-to-speech.
- All good: "You look good. I looked at your shirt, buttons and collar, and
  nothing looked wrong."
- Issue found: "I found one thing you may want to fix. A small dark mark, looks
  like food. Check the front of your shirt, near the collar."
"""

import json
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from litellm_utils import (
    extract_text,
    pil_image_to_data_uri,
    resolve_api_key,
    resolve_model_name,
)

# litellm is the shared LLM client used across ProgramAT tools. Import it
# defensively so the tool can return a clean spoken error instead of crashing
# if the package is missing.
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    litellm = None
    LITELLM_AVAILABLE = False

# IMPORTANT: this tool must never print() on the main() path. The streaming
# server captures stdout and PREPENDS it to the spoken audio (stream_server.py),
# so any debug print would be read aloud to the user. Surface problems through
# the returned dict instead.
#
# Privacy: this tool never writes the camera frame to disk (design doc section 13).

DARK_FRAME_THRESHOLD = 45.0     # avg grayscale brightness below this is too dark to judge
REQUEST_TIMEOUT = 30            # seconds per Gemini call; main() blocks the event loop
DETECT_TEMPERATURE = 0.2        # low - this is judgement, not creative writing
VERIFY_TEMPERATURE = 0.1        # lower still - verification is a yes/no fact check
MAX_IMAGE_SIZE = (1024, 1024)
JPEG_QUALITY = 90               # fine detail (stains) matters more here than payload size
GEMINI_DEFAULT_MODEL = 'gemini-3-flash-preview'

VALID_CATEGORIES = {
    'garment_orientation', 'closure', 'stain', 'wrinkle',
    'tag_or_label', 'collar_or_hem', 'debris', 'other',
}
_LEVELS = ('low', 'medium', 'high')


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resize_image_if_needed(image: np.ndarray, max_size: tuple = MAX_IMAGE_SIZE) -> np.ndarray:
    """
    Resize image efficiently while maintaining aspect ratio.

    Args:
        image: OpenCV image (numpy array)
        max_size: Maximum dimensions (width, height)

    Returns:
        Resized image if needed, original otherwise
    """
    height, width = image.shape[:2]
    max_width, max_height = max_size

    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def convert_cv2_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV BGR image to PIL RGB format.

    Args:
        image: OpenCV image (numpy array in BGR format)

    Returns:
        PIL Image in RGB format
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def average_brightness(image: np.ndarray) -> float:
    """
    Compute the mean grayscale brightness of a frame (0-255).

    This is a cheap, local, dependency-free check. It lets the tool catch an
    obviously-too-dark frame and give instant guidance without spending a
    Gemini call (design doc section 9.5 - tiered perception).

    Args:
        image: OpenCV image (numpy array, BGR or grayscale)

    Returns:
        Average brightness as a float; 0.0 for invalid input.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return 0.0
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(np.mean(gray))


# ---------------------------------------------------------------------------
# Prompts (kept as data - design doc section 9.6)
# ---------------------------------------------------------------------------

_DETECT_INSTRUCTIONS = """You are an appearance-check assistant for a blind or low-vision person who has pointed a phone camera at themselves. They cannot see the image or their own appearance. Your job is to honestly tell them whether anything is wrong with how they are dressed, so they can fix it before going out.

STEP 1 - Assess the photo first:
- Is a person visible, and how much of their body do you see (face_only, upper_body, full_body, or none)?
- Is the lighting good enough to judge clothing details such as stains or buttons?
- If the framing or lighting is NOT good enough to give a reliable answer, set capture.usable to false and write specific, human-sounding framing_guidance in the second person, for example: "Hold the phone a little farther away and tilt it down toward your chest." Never use coordinates, angles, or percentages.

STEP 2 - Only if the capture is usable, look for REAL problems in these categories: garment_orientation (clothing inside-out or backwards), closure (uneven buttons, an open zipper or fly), stain, wrinkle (only very obvious ones), tag_or_label (a tag sticking out), collar_or_hem (a folded collar, a shirt half-tucked), debris (lint, hair, or food stuck on). Never invent problems. Only report what you can actually see.

HONESTY RULES (research shows blind users distrust AI for this task, and false reassurance is the worst possible failure):
- NEVER give false reassurance. If you are unsure, lower the confidence and describe what you see neutrally, for example: "a darker area on the right sleeve; could be a stain or the fabric pattern".
- If the clothing has a busy pattern, say so and treat stain detection as less reliable.
- Give every location in body-relative terms the person can TOUCH, for example "the front of your shirt, near the waist" - never image coordinates.
- Say what an issue looks like (wet, dry, food, and so on) so they can confirm it by touch.
- In the "checked" list, name what you actually assessed, so the person knows what was and was not covered.
- Set overall to "looks_good" ONLY if the capture is usable AND there are no high or medium severity findings.

Return ONLY valid JSON, with no extra text, matching exactly this schema:
{
  "capture": {
    "person_visible": true or false,
    "body_coverage": "face_only" | "upper_body" | "full_body" | "none",
    "lighting": "good" | "dim" | "dark" | "bright",
    "usable": true or false,
    "framing_guidance": "" when usable, otherwise a spoken-style instruction telling the person how to retake the photo
  },
  "findings": [
    {
      "category": one of the STEP 2 category names,
      "description": "what is wrong, in plain words, including what it looks like",
      "location": "a body-relative location the person can touch",
      "severity": "high" | "medium" | "low",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "checked": ["short phrases naming what you assessed"],
  "overall": "looks_good" | "issues_found"
}
When there are no problems, "findings" must be an empty list."""


def build_detect_prompt(focus: Optional[str] = None) -> str:
    """
    Build the detection prompt for the first (broad) Gemini call.

    Args:
        focus: Optional area the person specifically asked about (design doc
            scenario 5). Empty in the shipped app, which sends no input_data.

    Returns:
        The full prompt string.
    """
    prompt = _DETECT_INSTRUCTIONS
    if focus and str(focus).strip():
        prompt += (
            f"\n\nThe person specifically asked you to focus on: {str(focus).strip()}. "
            "Pay extra attention there, but still report any other clear problems you notice."
        )
    return prompt


def build_verify_prompt(findings: List[Dict[str, Any]]) -> str:
    """
    Build the verification prompt for the second (narrow) Gemini call.

    The verify step re-checks each detected finding against the same image to
    cut false alarms (design doc section 9.3).

    Args:
        findings: The findings returned by the detection step.

    Returns:
        The full prompt string.
    """
    lines = []
    for index, finding in enumerate(findings, start=1):
        description = (str(finding.get('description', '')).strip() or 'a possible issue')
        location = (str(finding.get('location', '')).strip() or 'on the clothing')
        lines.append(f'{index}. "{description}" - at {location}')
    listing = '\n'.join(lines)

    return (
        'A previous appearance check of this same photo flagged the possible problems '
        'listed below. Look at the image again, and for EACH item look ONLY at that '
        'exact area. Confirm honestly whether the issue is really there. Do not be '
        'polite - a false alarm and a missed problem are both harmful.\n\n'
        f'{listing}\n\n'
        'Respond with ONLY a JSON array, one object per item, in the same order:\n'
        '[{"index": 1, "confirmed": true or false, "confidence": "high" | "medium" | '
        '"low", "note": "what you actually see in that area"}]\n'
        'If you cannot tell for an item, set confirmed to false and confidence to low.'
    )


# ---------------------------------------------------------------------------
# JSON parsing and normalization
# ---------------------------------------------------------------------------

def parse_json_object(text: str) -> Optional[Any]:
    """
    Parse a JSON object or array out of a model response, robustly.

    Handles a clean JSON string, a ```json fenced block, and JSON wrapped in
    surrounding prose.

    Args:
        text: Raw text from the model.

    Returns:
        The parsed object/array, or None if nothing valid could be extracted.
    """
    if not text or not isinstance(text, str):
        return None

    candidate = text.strip()

    # Strip a leading/trailing code fence (```json ... ``` or ``` ... ```).
    if candidate.startswith('```'):
        newline = candidate.find('\n')
        if newline != -1:
            candidate = candidate[newline + 1:]
        if candidate.rstrip().endswith('```'):
            candidate = candidate.rstrip()[:-3]
        candidate = candidate.strip()

    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        pass

    # Fall back to extracting the outermost {...} or [...] span.
    for open_char, close_char in (('{', '}'), ('[', ']')):
        start = candidate.find(open_char)
        end = candidate.rfind(close_char)
        if 0 <= start < end:
            try:
                return json.loads(candidate[start:end + 1])
            except (ValueError, TypeError):
                continue
    return None


def _norm_enum(value: Any, allowed, default: str) -> str:
    """Lowercase-normalize a string against an allowed set; fall back to default."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in allowed:
            return normalized
    return default


def _opt_bool(value: Any) -> Optional[bool]:
    """
    Parse a model-supplied boolean, returning None when it cannot be read.

    A plain bool() is unsafe on model JSON: bool("false") is True, because every
    non-empty string is truthy in Python - so a model that answered the string
    "false" would be misread as True. This recognizes the JSON literal and the
    common string and numeric spellings explicitly; anything else returns None
    so the caller can fail closed.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('true', 'yes', '1'):
            return True
        if normalized in ('false', 'no', '0'):
            return False
    if isinstance(value, (int, float)):
        return value != 0
    return None


def _rank(level: Any) -> int:
    """Map a low/medium/high level to 1/2/3. Unknown -> 2 (medium)."""
    try:
        return _LEVELS.index(str(level).strip().lower()) + 1
    except (ValueError, AttributeError):
        return 2


def _min_confidence(first: str, second: str) -> str:
    """Return the lower of two confidence levels (verify may only lower, not raise)."""
    return _LEVELS[min(_rank(first), _rank(second)) - 1]


def _normalize_finding(raw: Any) -> Optional[Dict[str, Any]]:
    """Coerce one raw finding into a clean, predictable dict (or None if unusable)."""
    if not isinstance(raw, dict):
        return None
    description = str(raw.get('description', '') or '').strip()
    if not description:
        return None  # nothing speakable - drop it
    return {
        'category': _norm_enum(raw.get('category'), VALID_CATEGORIES, 'other'),
        'description': description,
        'location': str(raw.get('location', '') or '').strip(),
        'severity': _norm_enum(raw.get('severity'), set(_LEVELS), 'medium'),
        'confidence': _norm_enum(raw.get('confidence'), set(_LEVELS), 'medium'),
    }


def _normalize_structured(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce the raw detection JSON into a clean, predictable structure.

    Crucially, this derives `capture.usable` defensively and fails closed: a
    missing person flag, a malformed or empty `capture` block, or any framing
    guidance all force `usable` false. An empty or junk model response (e.g.
    `{}`) therefore asks for a retake - it can never be mistaken for a clean
    "looks good".
    """
    raw_capture = data.get('capture') if isinstance(data.get('capture'), dict) else {}

    # Fail closed: a person counts as visible only when the model explicitly
    # says so. A plain bool() is unsafe here - it would default a missing flag
    # to True and read the string "false" as truthy.
    person_visible = _opt_bool(raw_capture.get('person_visible'))
    if person_visible is None:
        person_visible = False
    framing_guidance = str(raw_capture.get('framing_guidance', '') or '').strip()

    # Honor an explicit `usable` from the model (the string "false" included);
    # derive it only when the model omits it. The hard rules below always win:
    # no visible person, or any framing guidance, forces a retake.
    usable = _opt_bool(raw_capture.get('usable'))
    if usable is None:
        usable = person_visible and not framing_guidance
    if not person_visible or framing_guidance:
        usable = False

    capture = {
        'person_visible': person_visible,
        'body_coverage': _norm_enum(
            raw_capture.get('body_coverage'),
            {'face_only', 'upper_body', 'full_body', 'none'}, 'upper_body'),
        'lighting': _norm_enum(
            raw_capture.get('lighting'),
            {'good', 'dim', 'dark', 'bright'}, 'good'),
        'usable': usable,
        'framing_guidance': framing_guidance,
    }

    findings: List[Dict[str, Any]] = []
    raw_findings = data.get('findings')
    if isinstance(raw_findings, list):
        for raw_finding in raw_findings:
            normalized = _normalize_finding(raw_finding)
            if normalized:
                findings.append(normalized)

    checked: List[str] = []
    raw_checked = data.get('checked')
    if isinstance(raw_checked, list):
        checked = [str(item).strip() for item in raw_checked if str(item).strip()]

    return {
        'capture': capture,
        'findings': findings,
        'checked': checked,
        'overall': _norm_enum(
            data.get('overall'), {'looks_good', 'issues_found'}, 'issues_found'),
    }


# ---------------------------------------------------------------------------
# Gemini calls
# ---------------------------------------------------------------------------

def _vision_completion(model_name, api_key, prompt, image_data_uris, timeout,
                       temperature, json_object):
    """
    Run one multimodal LiteLLM completion (text + one or more images).

    `json_object` requests the provider's JSON-object mode (`response_format`).
    Pass it True only when the prompt asks for a JSON OBJECT; the verify prompt
    asks for a JSON ARRAY and passes False, because object mode fights an
    array-shaped prompt (some providers wrap or refuse the array) and the robust
    parse_json_object reads the array directly anyway.

    If the provider or LiteLLM version rejects `response_format` itself, the
    call is retried once without it. Any other error (auth, quota, network) is
    re-raised - retrying those would just fail again at double the latency and
    cost.
    """
    content: List[Dict[str, Any]] = [{'type': 'text', 'text': prompt}]
    for uri in image_data_uris:
        content.append({'type': 'image_url', 'image_url': {'url': uri}})

    base_kwargs = dict(
        model=model_name,
        messages=[{'role': 'user', 'content': content}],
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
    )

    if json_object:
        try:
            return litellm.completion(
                response_format={'type': 'json_object'}, **base_kwargs)
        except Exception as exc:
            # Retry plain only when response_format itself was rejected; an
            # auth/quota/network error would just fail again, so re-raise it.
            if 'response_format' not in str(exc).lower():
                raise
    return litellm.completion(**base_kwargs)


def analyze_appearance(image: np.ndarray, api_key: str, model_name: str,
                       focus: Optional[str] = None,
                       timeout: int = REQUEST_TIMEOUT) -> Dict[str, Any]:
    """
    Detection step: one Gemini call that assesses the capture and proposes findings.

    Args:
        image: OpenCV image (BGR numpy array).
        api_key: Provider API key.
        model_name: Model name (shorthand or fully qualified).
        focus: Optional area the person asked to focus on.
        timeout: Per-call timeout in seconds.

    Returns:
        {'success': True, 'structured': {...}} on success, or
        {'success': False, 'error': str} on any failure.
    """
    if not LITELLM_AVAILABLE:
        return {'success': False, 'error': 'litellm is not installed'}

    try:
        processed = resize_image_if_needed(image, max_size=MAX_IMAGE_SIZE)
        pil_image = convert_cv2_to_pil(processed)
        image_data_uri = pil_image_to_data_uri(pil_image, quality=JPEG_QUALITY)

        response = _vision_completion(
            resolve_model_name(model_name), api_key,
            build_detect_prompt(focus), [image_data_uri],
            timeout=timeout, temperature=DETECT_TEMPERATURE, json_object=True)

        parsed = parse_json_object(extract_text(response))
        if not isinstance(parsed, dict):
            return {'success': False, 'error': 'model response was not valid JSON'}

        return {'success': True, 'structured': _normalize_structured(parsed)}

    except Exception as exc:  # any failure -> caller speaks a safe retry message
        return {'success': False, 'error': str(exc)}


def _extract_verify_results(data: Any) -> Dict[int, Dict[str, Any]]:
    """Turn a verify response (array, or object wrapping one) into {index: result}."""
    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ('results', 'findings', 'verifications', 'items'):
            if isinstance(data.get(key), list):
                items = data[key]
                break

    results: Dict[int, Dict[str, Any]] = {}
    if not items:
        return results

    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get('index'))
        except (TypeError, ValueError):
            index = position + 1
        results[index] = item
    return results


def verify_findings(image: np.ndarray, findings: List[Dict[str, Any]], api_key: str,
                    model_name: str, timeout: int = REQUEST_TIMEOUT) -> List[Dict[str, Any]]:
    """
    Verification step: a second, focused Gemini call that re-checks each finding.

    Unconfirmed findings are dropped (false alarms); confirmed findings keep the
    LOWER of the detect and verify confidence - verification may only lower
    confidence, never inflate it. If the call fails for any reason, the original
    findings are returned unchanged (graceful degradation - never crash, never
    flip to a false "looks good").

    Args:
        image: OpenCV image (BGR numpy array).
        findings: Findings from analyze_appearance().
        api_key: Provider API key.
        model_name: Model name.
        timeout: Per-call timeout in seconds.

    Returns:
        The verified (possibly shorter) list of findings.
    """
    if not findings or not LITELLM_AVAILABLE:
        return findings

    try:
        processed = resize_image_if_needed(image, max_size=MAX_IMAGE_SIZE)
        pil_image = convert_cv2_to_pil(processed)
        image_data_uri = pil_image_to_data_uri(pil_image, quality=JPEG_QUALITY)

        response = _vision_completion(
            resolve_model_name(model_name), api_key,
            build_verify_prompt(findings), [image_data_uri],
            timeout=timeout, temperature=VERIFY_TEMPERATURE, json_object=False)

        results = _extract_verify_results(parse_json_object(extract_text(response)))
        if not results:
            return findings  # could not read the verification - trust detection

        verified: List[Dict[str, Any]] = []
        for position, finding in enumerate(findings):
            result = results.get(position + 1)
            if result is None:
                verified.append(finding)  # no verdict for this one - keep it
                continue

            confirmed = result.get('confirmed')
            is_explicitly_false = (
                confirmed is False
                or str(confirmed).strip().lower() == 'false')
            if is_explicitly_false:
                continue  # dropped as a false alarm

            verified_finding = dict(finding)
            verified_finding['confidence'] = _min_confidence(
                finding.get('confidence', 'medium'),
                _norm_enum(result.get('confidence'), set(_LEVELS),
                           finding.get('confidence', 'medium')))
            verified.append(verified_finding)
        return verified

    except Exception:
        return findings  # graceful degradation - keep the detection result


# ---------------------------------------------------------------------------
# Verdict and speech
# ---------------------------------------------------------------------------

def decide_overall(capture: Dict[str, Any], findings: List[Dict[str, Any]]) -> str:
    """
    Decide the final verdict. The tool, not the model, owns this (design doc 9.1).

    The safety invariant (design doc 9.6): "looks_good" is allowed ONLY when the
    capture was usable and nothing of medium or high severity remains after
    verification.
    """
    if not capture.get('usable', False):
        return 'issues_found'  # never claim "looks_good" on an unusable capture
    for finding in findings:
        if isinstance(finding, dict) and _rank(finding.get('severity')) >= 2:
            return 'issues_found'
    return 'looks_good'


def _format_checked(checked: Any, limit: int = 4) -> str:
    """Join the 'checked' phrases into a short, speakable list."""
    if not isinstance(checked, list):
        return ''
    items = [str(item).strip() for item in checked if str(item).strip()][:limit]
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def _describe_finding(finding: Dict[str, Any]) -> str:
    """Turn one finding into a spoken sentence: what, where, and how sure."""
    description = str(finding.get('description', '') or '').strip().rstrip('.')
    if not description:
        description = 'something looks off'
    description = description[0].upper() + description[1:]

    location = str(finding.get('location', '') or '').strip().rstrip('.')
    confidence = str(finding.get('confidence', 'medium')).strip().lower()

    sentence = description + '.'
    if location:
        sentence += f' Check {location}.'
    if confidence == 'low':
        # Honest uncertainty + a touchable verification hook (design doc 5.2).
        sentence += " I'm not certain about this one, so feel that spot to check, or ask someone."
    return sentence


def synthesize_speech(structured: Dict[str, Any]) -> str:
    """
    Turn the structured appearance result into concise, trust-calibrated speech.

    Output design (design doc section 12): for an unusable capture, speak only
    the framing guidance; for a clean result, give the verdict first and then
    what was checked (this builds trust); for issues, lead with the count, order
    by severity, give touchable locations, and hedge low-confidence findings.

    Args:
        structured: A normalized detection dict whose `overall` has already been
            set by decide_overall().

    Returns:
        The text to speak.
    """
    capture = structured.get('capture') if isinstance(structured.get('capture'), dict) else {}

    # Defensive: if called on an unusable capture, speak only the guidance.
    if not capture.get('usable', True):
        return capture.get('framing_guidance') or \
            "I couldn't see you well enough to check. Please try again."

    findings = [f for f in (structured.get('findings') or []) if isinstance(f, dict)]
    findings.sort(key=lambda f: _rank(f.get('severity')), reverse=True)
    checked_phrase = _format_checked(structured.get('checked', []))

    significant = [f for f in findings if _rank(f.get('severity')) >= 2]
    minor = [f for f in findings if _rank(f.get('severity')) < 2]

    if structured.get('overall') == 'looks_good':
        if not minor:
            parts = ['You look good.']
            if checked_phrase:
                parts.append(f'I looked at {checked_phrase}, and nothing looked wrong.')
            else:
                parts.append('Nothing looked wrong.')
            parts.append("You're all set.")
            return ' '.join(parts)

        parts = ['You look mostly good.', 'Just one small thing.',
                 _describe_finding(minor[0])]
        if checked_phrase:
            parts.append(f'Otherwise, I looked at {checked_phrase}, and that looked fine.')
        return ' '.join(parts)

    # overall == 'issues_found'
    if not significant and not minor:
        # Should not happen via decide_overall; stay safe, never reassure falsely.
        return ("I checked, but I couldn't get a clear enough read on your "
                "appearance. Please try again in better light.")

    count_word = {1: 'one thing', 2: 'two things', 3: 'three things'}.get(
        len(significant), 'a few things')
    if significant:
        parts = [f'I found {count_word} you may want to fix.']
        for finding in significant[:3]:
            parts.append(_describe_finding(finding))
    else:
        parts = ['A couple of small things to mention.']

    if minor and significant:
        parts.append('There is also something minor.')
        parts.append(_describe_finding(minor[0]))
    elif minor:
        for finding in minor[:2]:
            parts.append(_describe_finding(finding))

    if checked_phrase:
        parts.append(f'I also looked at {checked_phrase}, and that looked fine.')
    parts.append('Adjust those, and you can check again.')
    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Result builder and entry point
# ---------------------------------------------------------------------------

def _build_result(text: str, *, success: bool, audio_type: str, confidence: float,
                  structured: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the response dict in the shape the streaming server expects."""
    result: Dict[str, Any] = {
        'success': success,
        'description': text,
        'audio': {
            'type': audio_type,    # 'speech' for answers/guidance, 'error' for failures
            'text': text,
            'rate': 1.0,
            'interrupt': False,
        },
        'text': text,
        'confidence': confidence,
    }
    if structured is not None:
        # Ignored by the server; kept for a future Chat follow-up (design doc 9.2).
        result['structured'] = structured
    return result


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main entry point for the appearance check tool.
    Compatible with the ProgramAT tool execution framework.

    Pipeline: guard the image, do a cheap local lighting check, then run a
    detect -> verify pair of Gemini calls, let the tool decide the verdict, and
    speak a concise, trust-calibrated result.

    Args:
        image: Camera frame as a numpy array (BGR format from OpenCV).
        input_data: Optional configuration dictionary:
            - 'focus': Optional area to focus the check on (e.g. "the front of
              my shirt"). The shipped app sends no input_data.
            - 'verify': Set False to skip the verification call (default True).
            - 'api_key': Optional API key override.
            - 'model': Optional model override.

    Returns:
        Dictionary with the result and audio configuration:
        {
            'success': bool,
            'description': str,
            'audio': {'type': 'speech'|'error', 'text': str, 'rate': float,
                      'interrupt': bool},
            'text': str,
            'confidence': float,
            'structured': dict   # present when a structured result is available
        }
    """
    # 1. No usable image -> hard error (never a verdict).
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return _build_result(
            'No camera image is available to check your appearance.',
            success=False, audio_type='error', confidence=0.0)

    # 2. Normalize input_data. The shipped app sends {}, but a future targeted
    #    check (design doc scenario 5) may pass {'focus': '...'}.
    if not isinstance(input_data, dict):
        input_data = {}
    focus = input_data.get('focus')
    api_key = input_data.get('api_key')
    model = input_data.get(
        'model',
        os.environ.get('LLM_MODEL',
                       os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL)))
    do_verify = bool(input_data.get('verify', True))

    # 3. Cheap, local, dependency-free lighting check. Runs before any API call
    #    so an obviously-too-dark frame gets instant guidance and costs nothing.
    if average_brightness(image) < DARK_FRAME_THRESHOLD:
        return _build_result(
            "It's too dark for me to check how you look. Turn on a light or "
            'move near a window, then check again.',
            success=True, audio_type='speech', confidence=0.5)

    # 4. The perception step needs litellm and a configured key.
    if not LITELLM_AVAILABLE:
        return _build_result(
            "The appearance check can't run because the litellm package is "
            'not installed.',
            success=False, audio_type='error', confidence=0.0)
    api_key = resolve_api_key(model, api_key or '')
    if not api_key:
        return _build_result(
            "The appearance check isn't configured yet. The Gemini API key "
            'is missing.',
            success=False, audio_type='error', confidence=0.0)

    # 5. Detect: one Gemini call -> capture assessment + proposed findings.
    detection = analyze_appearance(
        image, api_key=api_key, model_name=model, focus=focus,
        timeout=REQUEST_TIMEOUT)
    if not detection.get('success'):
        return _build_result(
            "I couldn't check your appearance just now. Please try again.",
            success=False, audio_type='error', confidence=0.0)
    structured = detection['structured']
    capture = structured.get('capture', {})

    # 6. Bad framing or lighting -> speak guidance, never a verdict (design doc 6).
    if not capture.get('usable', False):
        guidance = capture.get('framing_guidance') or (
            "I can't see you well enough. Hold the phone farther away, "
            'pointed at your chest.')
        if 'again' not in guidance.lower():
            guidance = guidance.rstrip() + ' Then check again.'
        return _build_result(
            guidance, success=True, audio_type='speech', confidence=0.5,
            structured=structured)

    # 7. Verify: a focused second look re-checks each finding and drops false
    #    alarms (design doc 9.3). Conditional - skipped when nothing was found.
    findings = structured.get('findings', [])
    if do_verify and findings:
        findings = verify_findings(
            image, findings, api_key=api_key, model_name=model,
            timeout=REQUEST_TIMEOUT)
    structured['findings'] = findings

    # 8. The tool - not the model - owns the final verdict (design doc 9.1, 9.6).
    structured['overall'] = decide_overall(capture, findings)

    # 9. Speak it.
    speech = synthesize_speech(structured)
    return _build_result(
        speech, success=True, audio_type='speech', confidence=0.9,
        structured=structured)


# Building block exports for use by other tools and tests.
__all__ = [
    'main',
    'analyze_appearance',
    'verify_findings',
    'synthesize_speech',
    'decide_overall',
    'build_detect_prompt',
    'build_verify_prompt',
    'parse_json_object',
    'average_brightness',
    'resize_image_if_needed',
    'convert_cv2_to_pil',
]
