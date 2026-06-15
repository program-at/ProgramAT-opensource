"""
Targeted tests for tools/grocery_product_price_identifier.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import grocery_product_price_identifier as tool


def test_parsing_helpers():
    sample = "Kirkland Organic Milk\n2% Reduced Fat\n$3.49"
    parsed = tool.parse_product_and_price(sample, 'bottle')
    assert parsed['name'] == 'Kirkland Organic Milk'
    assert parsed['price'] == '$3.49'

    no_price = tool.parse_product_and_price("Fresh Apples", 'apple')
    assert no_price['name'] == 'Fresh Apples'
    assert no_price['price'] is None


def test_main_with_mocks():
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255

    tool.yolo_model_cache = {}

    original_detect = tool.detect_products
    original_extract = tool.extract_text_from_region

    try:
        tool.detect_products = lambda *_args, **_kwargs: [
            {
                'class_name': 'bottle',
                'confidence': 0.92,
                'bbox': [120, 80, 320, 360],
                'center': [280, 260],
            }
        ]
        tool.extract_text_from_region = lambda *_args, **_kwargs: "Kirkland Water\n$1.99"

        first = tool.main(image, {'track_mode': True})
        assert isinstance(first, dict)
        assert 'Kirkland Water' in first['text']
        assert '$1.99' in first['text']

        second = tool.main(image, {'track_mode': True})
        assert second == ""

        one_shot = tool.main(image, {'track_mode': False})
        assert isinstance(one_shot, dict)
        assert 'price $1.99' in one_shot['text']
    finally:
        tool.detect_products = original_detect
        tool.extract_text_from_region = original_extract


if __name__ == '__main__':
    test_parsing_helpers()
    test_main_with_mocks()
    print('All grocery product-price tests passed.')
