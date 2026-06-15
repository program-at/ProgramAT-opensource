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

    whole_dollar = tool.parse_product_and_price("Bread Special $5", 'sandwich')
    assert whole_dollar['price'] == '$5.00'


def test_store_item_scope_defaults():
    assert tool.DEFAULT_CONFIDENCE == 0.25
    assert 'COCO_CLASSES' not in tool.__dict__
    assert 'STORE_ITEM_CLASSES' not in tool.__dict__

    label_first = "Big Promo Header\nStore Brand Chips\n$2.49\n2 for 4"
    parsed = tool.parse_product_and_price(label_first, 'object_0')
    assert parsed['name'] == 'Store Brand Chips'
    assert parsed['price'] == '$2.49'


def test_main_with_mocks():
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255

    if 'yolo_model_cache' not in tool.__dict__:
        tool.__dict__['yolo_model_cache'] = {}
    tool._get_shared_cache().clear()

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


def test_detect_products_uses_model_label_names():
    image = np.ones((64, 64, 3), dtype=np.uint8)

    class _FakeTensor:
        def __init__(self, values):
            self._values = np.array(values, dtype=float)

        def __getitem__(self, index):
            return _FakeTensor(self._values[index])

        def cpu(self):
            return self

        def numpy(self):
            return self._values

        def __int__(self):
            return int(self._values.item())

        def __float__(self):
            return float(self._values.item())

    class _FakeBox:
        def __init__(self):
            self.cls = _FakeTensor([0])
            self.conf = _FakeTensor([0.88])
            self.xyxy = _FakeTensor([[4, 6, 28, 32]])

    class _FakeResult:
        def __init__(self):
            self.boxes = [_FakeBox()]
            self.names = {0: "snack package"}

    class _FakeModel:
        def __call__(self, *_args, **_kwargs):
            return [_FakeResult()]

    original_loader = tool._get_or_load_yolo_model
    try:
        tool._get_or_load_yolo_model = lambda: _FakeModel()
        detections = tool.detect_products(image, confidence_threshold=0.1)
        assert len(detections) == 1
        assert detections[0]['class_name'] == "snack package"
    finally:
        tool._get_or_load_yolo_model = original_loader


if __name__ == '__main__':
    test_parsing_helpers()
    test_store_item_scope_defaults()
    test_main_with_mocks()
    test_detect_products_uses_model_label_names()
    print('All store product-price tests passed.')
