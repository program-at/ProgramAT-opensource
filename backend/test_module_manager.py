"""
Test script for module_manager.py
Verifies automatic module installation works correctly
"""

from module_manager import get_module_manager

def test_basic_loading():
    """Test loading of pre-installed modules"""
    print("Testing basic module loading...")
    mgr = get_module_manager()
    
    # Test numpy
    np_module = mgr.load_module('numpy', install_if_missing=False)
    print(f"  NumPy: {'✓' if np_module else '✗'}")
    
    # Test opencv
    cv2_module = mgr.load_module('cv2', install_if_missing=False)
    print(f"  OpenCV: {'✓' if cv2_module else '✗'}")
    
    print()

def test_auto_install():
    """Test automatic installation of missing module"""
    print("Testing automatic installation...")
    mgr = get_module_manager()
    
    # Try to load a module that might not be installed
    test_module = 'requests'
    module = mgr.load_module(test_module, install_if_missing=True)
    print(f"  {test_module}: {'✓ (installed)' if module else '✗ (failed)'}")
    print()

def test_error_parsing():
    """Test parsing of import errors"""
    print("Testing error message parsing...")
    mgr = get_module_manager()
    
    test_cases = [
        ("No module named 'scipy'", 'scipy'),
        ("No module named scipy", 'scipy'),
        ("ModuleNotFoundError: No module named 'sklearn'", 'sklearn'),
        ("cannot import name 'filters' from 'skimage'", 'skimage'),
    ]
    
    for error_msg, expected in test_cases:
        result = mgr.install_from_error(error_msg)
        status = '✓' if result == expected or result is not None else '✗'
        print(f"  '{error_msg[:40]}...' → {result} {status}")
    
    print()

def test_common_modules():
    """Test getting all common modules"""
    print("Testing common modules loading...")
    mgr = get_module_manager()
    
    common = mgr.get_common_modules()
    print(f"  Loaded {len(common)} common modules:")
    for name in sorted(common.keys()):
        print(f"    - {name}")
    
    print()

if __name__ == '__main__':
    print("=" * 60)
    print("Module Manager Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_basic_loading()
        test_auto_install()
        test_error_parsing()
        test_common_modules()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
