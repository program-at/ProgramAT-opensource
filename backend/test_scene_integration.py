#!/usr/bin/env python3
"""
Integration test for scene_description tool with backend patterns
Tests that the tool works as it would when called from stream_server.py
"""

import sys
import os
import cv2
import numpy as np
import json

# Add tools and backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
sys.path.insert(0, os.path.dirname(__file__))

def simulate_backend_execution():
    """
    Simulate how the backend would execute the tool
    This mimics the pattern in stream_server.py
    """
    print("=" * 60)
    print("Integration Test: Backend Execution Pattern")
    print("=" * 60)
    
    # Import the tool
    import scene_description
    
    # Create a test image (simulating camera frame)
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Gray background
    
    # Add some visual elements
    cv2.rectangle(test_img, (100, 100), (300, 300), (0, 255, 0), -1)
    cv2.circle(test_img, (450, 200), 50, (255, 0, 0), -1)
    cv2.putText(test_img, 'HELLO', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # Test different input configurations
    test_cases = [
        {
            'name': 'Default configuration',
            'input_data': {}
        },
        {
            'name': 'Brief and concise',
            'input_data': {
                'detail_level': 'brief',
                'style': 'concise'
            }
        },
        {
            'name': 'Detailed with text focus',
            'input_data': {
                'detail_level': 'detailed',
                'focus': 'text',
                'style': 'narrative'
            }
        },
        {
            'name': 'Navigation focus',
            'input_data': {
                'detail_level': 'standard',
                'focus': 'navigation'
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        # Execute the tool (like backend does)
        try:
            result = scene_description.main(test_img, test_case['input_data'])
            
            # Verify result structure (what backend expects)
            assert isinstance(result, dict), "Result must be a dictionary"
            
            # Check for audio field (required for TTS)
            if 'audio' in result:
                audio = result['audio']
                assert 'type' in audio, "Audio must have 'type' field"
                assert 'text' in audio, "Audio must have 'text' field"
                print(f"✓ Audio type: {audio['type']}")
                print(f"✓ Audio text: {audio['text'][:80]}{'...' if len(audio['text']) > 80 else ''}")
            else:
                # Some tools might return simple strings
                print(f"⚠ No audio field - result: {str(result)[:80]}...")
            
            # Check for text field (for display)
            if 'text' in result:
                print(f"✓ Display text: {result['text'][:60]}{'...' if len(result['text']) > 60 else ''}")
            
            # Check additional metadata
            if 'success' in result:
                print(f"✓ Success: {result['success']}")
            
            if 'confidence' in result:
                print(f"✓ Confidence: {result['confidence']:.2f}")
            
            if 'context' in result:
                print(f"✓ Context available: {list(result['context'].keys())}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_error_conditions():
    """Test that the tool handles error conditions properly"""
    print("\n" + "=" * 60)
    print("Integration Test: Error Handling")
    print("=" * 60)
    
    import scene_description
    
    # Test 1: None image
    print("\n--- Test: None image ---")
    result = scene_description.main(None, {})
    assert result['audio']['type'] == 'error', "Should return error type for None image"
    assert 'No camera image' in result['audio']['text'], "Should mention no camera image"
    print("✓ Handles None image correctly")
    
    # Test 2: Empty image
    print("\n--- Test: Empty image ---")
    empty_img = np.array([])
    try:
        result = scene_description.main(empty_img, {})
        # Should handle gracefully
        print(f"✓ Handles empty image: {result.get('audio', {}).get('type', 'unknown')}")
    except Exception as e:
        print(f"⚠ Empty image raises exception: {e}")
    
    # Test 3: No API key (already covered in main tests, but verify error message)
    print("\n--- Test: No API key ---")
    old_key = os.environ.pop('GEMINI_API_KEY', None)
    test_img = np.ones((100, 100, 3), dtype=np.uint8)
    result = scene_description.main(test_img, {})
    assert result['success'] == False, "Should fail without API key"
    assert 'GEMINI_API_KEY' in result['description'], "Should mention API key in error"
    print("✓ Provides clear error message about missing API key")
    if old_key:
        os.environ['GEMINI_API_KEY'] = old_key
    
    return True

def test_module_imports():
    """Test that the tool can be imported by other tools"""
    print("\n" + "=" * 60)
    print("Integration Test: Module Imports (Building Blocks)")
    print("=" * 60)
    
    # Test importing individual functions
    try:
        from scene_description import (
            analyze_scene,
            format_description_for_audio,
            get_scene_context,
            build_scene_prompt
        )
        print("✓ Successfully imported building block functions")
        
        # Quick test of each
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        context = get_scene_context(test_img)
        assert 'width' in context
        print(f"✓ get_scene_context works: {context['resolution']}")
        
        prompt = build_scene_prompt('brief', 'general')
        assert len(prompt) > 0
        print(f"✓ build_scene_prompt works: {len(prompt)} chars")
        
        formatted = format_description_for_audio("This is a test description.", 'concise')
        assert len(formatted) > 0
        print(f"✓ format_description_for_audio works")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False

def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("SCENE DESCRIPTION TOOL - INTEGRATION TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        ("Backend Execution Pattern", simulate_backend_execution),
        ("Error Handling", test_error_conditions),
        ("Module Imports", test_module_imports),
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
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} integration tests passed")
    
    return 0 if passed_count == total_count else 1

if __name__ == '__main__':
    sys.exit(main())
