#!/usr/bin/env python3
"""
Test script for scene_description tool
Creates a simple test image and verifies the tool's functionality
"""

import sys
import os
import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

def create_test_image():
    """Create a simple test image for verification"""
    # Create a 640x480 image with some content
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
    
    # Add some colored rectangles
    cv2.rectangle(img, (50, 50), (250, 250), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(img, (300, 50), (500, 250), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(img, (175, 300), (375, 450), (0, 0, 255), -1)  # Red rectangle
    
    # Add some text
    cv2.putText(img, 'TEST IMAGE', (200, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def test_without_api_key():
    """Test the tool without an API key (should return error message)"""
    print("=" * 60)
    print("Test 1: Without API Key (Expected to show error)")
    print("=" * 60)
    
    import scene_description
    
    # Create test image
    test_img = create_test_image()
    
    # Remove API key from environment if present
    old_key = os.environ.pop('GEMINI_API_KEY', None)
    
    # Run tool
    result = scene_description.main(test_img, {})
    
    # Restore API key
    if old_key:
        os.environ['GEMINI_API_KEY'] = old_key
    
    print(f"Success: {result.get('success')}")
    print(f"Description: {result.get('description')}")
    print(f"Audio type: {result.get('audio', {}).get('type')}")
    print(f"Audio text: {result.get('audio', {}).get('text')}")
    print()
    
    return result.get('success') == False

def test_building_blocks():
    """Test building block functions"""
    print("=" * 60)
    print("Test 2: Building Block Functions")
    print("=" * 60)
    
    import scene_description
    
    # Create test image
    test_img = create_test_image()
    
    # Test get_scene_context
    context = scene_description.get_scene_context(test_img)
    print(f"Image context: {context}")
    assert context['width'] == 640
    assert context['height'] == 480
    print("✓ get_scene_context works")
    
    # Test resize_image_if_needed
    large_img = np.ones((2000, 3000, 3), dtype=np.uint8)
    resized = scene_description.resize_image_if_needed(large_img, max_size=(1024, 1024))
    print(f"Original size: 3000x2000, Resized: {resized.shape[1]}x{resized.shape[0]}")
    assert resized.shape[0] <= 1024 and resized.shape[1] <= 1024
    print("✓ resize_image_if_needed works")
    
    # Test format_description_for_audio
    long_desc = "In this image, there are many wonderful things to see. " * 20
    formatted = scene_description.format_description_for_audio(long_desc, style='concise', max_length=100)
    print(f"Formatted description length: {len(formatted)} (max: 100)")
    assert len(formatted) <= 100
    print("✓ format_description_for_audio works")
    
    # Test build_scene_prompt
    prompt = scene_description.build_scene_prompt(detail_level='detailed', focus='navigation')
    print(f"Generated prompt length: {len(prompt)} characters")
    assert 'navigation' in prompt.lower()
    print("✓ build_scene_prompt works")
    
    print()
    return True

def test_with_api_key():
    """Test with API key if available"""
    print("=" * 60)
    print("Test 3: With API Key (if available)")
    print("=" * 60)
    
    api_key = os.environ.get('GEMINI_API_KEY', '')
    
    if not api_key:
        print("⚠ GEMINI_API_KEY not set, skipping API test")
        print("  To test with Gemini API, set: export GEMINI_API_KEY=your_key")
        return True
    
    import scene_description
    
    # Create test image
    test_img = create_test_image()
    
    # Test different configurations
    configs = [
        {'detail_level': 'brief', 'focus': 'general', 'style': 'concise'},
        {'detail_level': 'standard', 'focus': 'objects', 'style': 'narrative'},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        result = scene_description.main(test_img, config)
        
        print(f"Success: {result.get('success')}")
        print(f"Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"Description: {result.get('description', '')[:100]}...")
        
        if not result.get('success'):
            print(f"Error: {result.get('description')}")
    
    print()
    return True

def test_no_image():
    """Test with None image"""
    print("=" * 60)
    print("Test 4: No Image (Expected to handle gracefully)")
    print("=" * 60)
    
    import scene_description
    
    result = scene_description.main(None, {})
    
    print(f"Success: {result.get('success')}")
    print(f"Audio type: {result.get('audio', {}).get('type')}")
    print(f"Audio text: {result.get('audio', {}).get('text')}")
    print()
    
    return result.get('success') == False and result.get('audio', {}).get('type') == 'error'

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("SCENE DESCRIPTION TOOL - TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        ("No API Key Error Handling", test_without_api_key),
        ("Building Block Functions", test_building_blocks),
        ("API Integration (if key available)", test_with_api_key),
        ("No Image Error Handling", test_no_image),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    return 0 if passed_count == total_count else 1

if __name__ == '__main__':
    sys.exit(main())
