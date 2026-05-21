"""
Test script for appearance_check.py

Exercises the appearance check tool. The building-block tests run fully offline
(no API key needed); the live test is skipped automatically unless GEMINI_API_KEY
is set in the environment.

Run:  python backend/test_appearance_check.py
"""

import os
import sys

import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from appearance_check import (
    average_brightness,
    build_detect_prompt,
    build_verify_prompt,
    decide_overall,
    main,
    parse_json_object,
    resize_image_if_needed,
    synthesize_speech,
    _describe_finding,
    _normalize_structured,
)


def create_test_image():
    """Create a bright test image with colored shapes simulating clothing."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 235
    cv2.rectangle(image, (180, 60), (460, 360), (90, 60, 200), -1)   # a "shirt"
    cv2.rectangle(image, (300, 140), (340, 320), (250, 250, 250), -1)  # a placket
    return image


def create_dark_image():
    """Create an almost-black frame (simulates a lights-off / covered lens shot)."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 15


def test_average_brightness():
    """Brightness measurement separates a dark frame from a bright one."""
    print("Testing average_brightness()...")
    dark = average_brightness(create_dark_image())
    bright = average_brightness(create_test_image())
    print(f"  Dark frame brightness:   {dark:.1f}")
    print(f"  Bright frame brightness: {bright:.1f}")
    assert dark < 45, f"Dark frame should be below 45, got {dark}"
    assert bright > 200, f"Bright frame should be above 200, got {bright}"
    assert average_brightness(None) == 0.0, "None image should be 0.0"
    print("  PASS\n")


def test_resize_image():
    """Resizing shrinks oversized frames and leaves small ones untouched."""
    print("Testing resize_image_if_needed()...")
    image = create_test_image()
    resized = resize_image_if_needed(image, max_size=(320, 240))
    not_resized = resize_image_if_needed(image, max_size=(1920, 1080))
    print(f"  Original {image.shape} -> downscaled {resized.shape}")
    assert resized.shape[1] <= 320 and resized.shape[0] <= 240
    assert not_resized.shape == image.shape
    print("  PASS\n")


def test_build_detect_prompt():
    """The detect prompt carries the schema and honesty rules; focus is appended."""
    print("Testing build_detect_prompt()...")
    prompt = build_detect_prompt()
    for token in ('capture', 'findings', 'usable', 'looks_good',
                  'framing_guidance', 'JSON'):
        assert token in prompt, f"detect prompt missing '{token}'"
    focused = build_detect_prompt('the front of my shirt')
    assert 'the front of my shirt' in focused, "focus text not appended"
    print(f"  Base prompt: {len(prompt)} chars; focused prompt adds the focus area")
    print("  PASS\n")


def test_build_verify_prompt():
    """The verify prompt lists each finding and asks for a JSON array."""
    print("Testing build_verify_prompt()...")
    findings = [
        {'description': 'dark mark, looks like food', 'location': 'front of shirt'},
        {'description': 'a button is undone', 'location': 'mid-chest'},
    ]
    prompt = build_verify_prompt(findings)
    assert 'dark mark, looks like food' in prompt
    assert 'a button is undone' in prompt
    for token in ('confirmed', 'index', 'JSON array'):
        assert token in prompt, f"verify prompt missing '{token}'"
    print(f"  Verify prompt lists {len(findings)} findings, asks for a JSON array")
    print("  PASS\n")


def test_parse_json_object():
    """JSON is recovered from plain, fenced, prose-wrapped, and array responses."""
    print("Testing parse_json_object()...")
    assert parse_json_object('{"a": 1}') == {'a': 1}
    assert parse_json_object('```json\n{"a": 2}\n```') == {'a': 2}
    assert parse_json_object('```\n{"a": 3}\n```') == {'a': 3}
    assert parse_json_object('Sure, here it is: {"a": 4} hope that helps') == {'a': 4}
    assert parse_json_object('[1, 2, 3]') == [1, 2, 3]
    assert parse_json_object('no json here at all') is None
    assert parse_json_object('') is None
    assert parse_json_object(None) is None
    print("  Plain, fenced, prose-wrapped, array, and garbage all handled")
    print("  PASS\n")


def test_normalize_structured():
    """Messy model JSON is coerced to a clean, predictable structure."""
    print("Testing _normalize_structured()...")
    raw = {
        'capture': {'person_visible': True, 'usable': True, 'lighting': 'GOOD',
                    'body_coverage': 'Upper_Body', 'framing_guidance': ''},
        'findings': [
            {'category': 'STAIN', 'description': 'dark mark', 'location': 'front',
             'severity': 'HIGH', 'confidence': 'medium'},
            {'category': 'closure', 'description': '', 'location': 'x'},  # empty -> dropped
            'not a dict',                                                # -> dropped
        ],
        'checked': ['buttons', '   ', 'collar'],
        'overall': 'ISSUES_FOUND',
    }
    result = _normalize_structured(raw)
    assert result['capture']['lighting'] == 'good'
    assert result['capture']['body_coverage'] == 'upper_body'
    assert len(result['findings']) == 1, "empty-description / non-dict findings must drop"
    assert result['findings'][0]['category'] == 'stain'
    assert result['findings'][0]['severity'] == 'high'
    assert result['checked'] == ['buttons', 'collar'], "blank checked items must drop"
    assert result['overall'] == 'issues_found'

    # usable is inferred when the model omits it.
    inferred_ok = _normalize_structured({'capture': {'person_visible': True}})
    assert inferred_ok['capture']['usable'] is True, "visible + no guidance -> usable"
    inferred_bad = _normalize_structured(
        {'capture': {'person_visible': True, 'framing_guidance': 'Move closer.'}})
    assert inferred_bad['capture']['usable'] is False, "framing guidance -> not usable"
    no_person = _normalize_structured({'capture': {'person_visible': False, 'usable': True}})
    assert no_person['capture']['usable'] is False, "no person -> never usable"
    print("  Enums lowercased, junk findings dropped, usable inferred safely")
    print("  PASS\n")


def test_decide_overall():
    """The tool owns the verdict; 'looks_good' needs a usable, clean result."""
    print("Testing decide_overall()...")
    assert decide_overall({'usable': True}, [{'severity': 'high'}]) == 'issues_found'
    assert decide_overall({'usable': True}, [{'severity': 'medium'}]) == 'issues_found'
    assert decide_overall({'usable': True}, [{'severity': 'low'}]) == 'looks_good'
    assert decide_overall({'usable': True}, []) == 'looks_good'
    assert decide_overall({'usable': False}, []) == 'issues_found', \
        "an unusable capture must never read as looks_good"
    print("  High/medium -> issues_found; low-only and clean -> looks_good")
    print("  PASS\n")


def test_synthesize_speech():
    """Speech is verdict-first, lists what was checked, and never reassures falsely."""
    print("Testing synthesize_speech()...")

    clean = synthesize_speech({
        'capture': {'usable': True},
        'findings': [],
        'checked': ['shirt orientation', 'buttons', 'collar'],
        'overall': 'looks_good',
    })
    print(f"  looks_good : {clean}")
    assert 'look good' in clean.lower()
    assert 'buttons' in clean

    minor = synthesize_speech({
        'capture': {'usable': True},
        'findings': [{'description': 'a slight wrinkle', 'location': 'your right sleeve',
                      'severity': 'low', 'confidence': 'medium'}],
        'checked': ['buttons'],
        'overall': 'looks_good',
    })
    print(f"  minor note : {minor}")
    assert 'mostly good' in minor.lower()
    assert 'wrinkle' in minor.lower()

    issue = synthesize_speech({
        'capture': {'usable': True},
        'findings': [{'description': 'your shirt is inside-out',
                      'location': 'the collar seam', 'severity': 'high',
                      'confidence': 'high'}],
        'checked': ['buttons', 'stains'],
        'overall': 'issues_found',
    })
    print(f"  issue      : {issue}")
    assert 'inside-out' in issue.lower()
    assert 'collar seam' in issue
    # Safety invariant: an issue result must never contain a false "you look good".
    assert 'you look good' not in issue.lower(), "false reassurance on an issue result"

    guidance = synthesize_speech({
        'capture': {'usable': False, 'framing_guidance': 'Move into better light.'},
    })
    print(f"  guidance   : {guidance}")
    assert guidance == 'Move into better light.', "unusable capture must speak only guidance"

    # Low-confidence findings are hedged and defer to touch / another person.
    hedged = _describe_finding({'description': 'a darker patch', 'location': 'your cuff',
                                'confidence': 'low'})
    assert 'not certain' in hedged.lower()
    print("  PASS\n")


def test_main_no_image():
    """main() with no usable image returns a hard error, never a verdict."""
    print("Testing main() with no image...")
    for bad in (None, np.array([]), 'not an image'):
        result = main(bad)
        assert result['success'] is False
        assert result['audio']['type'] == 'error'
        assert result['text'], "an error result still needs text"
    print("  None, empty array, and non-array all return a clean error")
    print("  PASS\n")


def test_main_dark_image():
    """main() on a too-dark frame gives lighting guidance offline (no API call)."""
    print("Testing main() with a dark frame (offline)...")
    result = main(create_dark_image())
    print(f"  Spoken: {result['text']}")
    assert result['audio']['type'] == 'speech', "lighting guidance should speak"
    assert 'dark' in result['text'].lower() or 'light' in result['text'].lower()
    assert 'you look good' not in result['text'].lower(), "no false reassurance"
    print("  Too-dark frame short-circuits to lighting guidance, no API needed")
    print("  PASS\n")


def test_main_live():
    """Live end-to-end test against Gemini (requires GEMINI_API_KEY)."""
    print("Testing main() live against Gemini...")
    if not os.environ.get('GEMINI_API_KEY'):
        print("  GEMINI_API_KEY not set, skipping live API test")
        print("  Set GEMINI_API_KEY to run the live end-to-end check")
        print("  SKIPPED\n")
        return

    result = main(create_test_image())
    print(f"  success    : {result.get('success')}")
    print(f"  audio type : {result.get('audio', {}).get('type')}")
    print(f"  spoken     : {result.get('text')}")
    assert isinstance(result, dict)
    assert 'audio' in result and 'text' in result
    assert result['audio']['type'] in ('speech', 'error')
    assert result['text'], "live result must have spoken text"
    print("  Live call returned a well-formed result")
    print("  PASS\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("APPEARANCE CHECK TOOL - TEST SUITE")
    print("=" * 60)
    print()

    # Offline building-block tests.
    test_average_brightness()
    test_resize_image()
    test_build_detect_prompt()
    test_build_verify_prompt()
    test_parse_json_object()
    test_normalize_structured()
    test_decide_overall()
    test_synthesize_speech()
    test_main_no_image()
    test_main_dark_image()

    print("-" * 60)
    print("ALL OFFLINE TESTS PASSED")
    print("-" * 60)
    print()

    # Live test (auto-skips without a key).
    test_main_live()

    print("=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
