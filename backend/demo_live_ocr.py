"""
Live OCR Tool Demo
Demonstrates the live OCR tool functionality without requiring actual OCR API
"""

import sys
import os

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from live_ocr import (
    calculate_text_similarity,
    chunk_text_for_audio,
    format_text_for_speech,
    reset_text_tracking,
    update_text_history,
    is_text_similar_to_history,
)


def demo_text_tracking():
    """Demonstrate text tracking and similarity detection"""
    print("=" * 70)
    print("LIVE OCR DEMONSTRATION - Text Tracking Feature")
    print("=" * 70)
    print()
    
    # Scenario: User scanning a milk bottle, then nutrition facts, then newspaper
    
    print("Scenario: User scanning different objects with camera")
    print("-" * 70)
    print()
    
    # Reset tracking
    reset_text_tracking()
    print("1. Starting fresh (text tracking reset)\n")
    
    # First text: Milk bottle label
    text1 = "Milk 2% Kirkland"
    print(f"2. Camera sees: '{text1}'")
    is_duplicate = is_text_similar_to_history(text1)
    print(f"   Is this duplicate? {is_duplicate}")
    if not is_duplicate:
        update_text_history(text1)
        chunks = chunk_text_for_audio(text1, chunk_size=5)
        speech = format_text_for_speech(', '.join(chunks))
        print(f"   📢 SPEAK: '{speech}'")
    else:
        print("   🔇 SILENT (duplicate)")
    print()
    
    # Same text again (should be silent)
    text2 = "Milk 2% Kirkland"
    print(f"3. Camera still sees: '{text2}'")
    is_duplicate = is_text_similar_to_history(text2, threshold=0.8)
    print(f"   Is this duplicate? {is_duplicate}")
    similarity = calculate_text_similarity(text1, text2)
    print(f"   Similarity to previous: {similarity:.2f}")
    if not is_duplicate:
        update_text_history(text2)
        chunks = chunk_text_for_audio(text2, chunk_size=5)
        speech = format_text_for_speech(', '.join(chunks))
        print(f"   📢 SPEAK: '{speech}'")
    else:
        print("   🔇 SILENT (duplicate)")
    print()
    
    # Turn bottle to nutrition facts
    text3 = "Servings per bottle 10 Calories 50 Fat 3% Carbs 12% Calcium 30%"
    print(f"4. User turns bottle, camera sees: '{text3}'")
    is_duplicate = is_text_similar_to_history(text3, threshold=0.8)
    print(f"   Is this duplicate? {is_duplicate}")
    similarity_to_prev = calculate_text_similarity(text2, text3)
    print(f"   Similarity to previous: {similarity_to_prev:.2f}")
    if not is_duplicate:
        update_text_history(text3)
        chunks = chunk_text_for_audio(text3, chunk_size=8)
        speech = format_text_for_speech(', '.join(chunks))
        print(f"   📢 SPEAK: '{speech}'")
    else:
        print("   🔇 SILENT (duplicate)")
    print()
    
    # Move to newspaper
    text4 = "New York Times January 14 2026 Headline Breaking News"
    print(f"5. User moves to newspaper, camera sees: '{text4}'")
    is_duplicate = is_text_similar_to_history(text4, threshold=0.8)
    print(f"   Is this duplicate? {is_duplicate}")
    similarity_to_nutrition = calculate_text_similarity(text3, text4)
    print(f"   Similarity to nutrition facts: {similarity_to_nutrition:.2f}")
    if not is_duplicate:
        update_text_history(text4)
        chunks = chunk_text_for_audio(text4, chunk_size=8)
        speech = format_text_for_speech(', '.join(chunks))
        print(f"   📢 SPEAK: '{speech}'")
    else:
        print("   🔇 SILENT (duplicate)")
    print()
    
    # Stay on newspaper (should be silent)
    text5 = "New York Times January 14 2026 Headline Breaking News"
    print(f"6. Camera still on newspaper: '{text5}'")
    is_duplicate = is_text_similar_to_history(text5, threshold=0.8)
    print(f"   Is this duplicate? {is_duplicate}")
    similarity_to_prev_newspaper = calculate_text_similarity(text4, text5)
    print(f"   Similarity to previous reading: {similarity_to_prev_newspaper:.2f}")
    if not is_duplicate:
        update_text_history(text5)
        chunks = chunk_text_for_audio(text5, chunk_size=8)
        speech = format_text_for_speech(', '.join(chunks))
        print(f"   📢 SPEAK: '{speech}'")
    else:
        print("   🔇 SILENT (duplicate)")
    print()
    
    print("-" * 70)
    print()


def demo_chunking():
    """Demonstrate text chunking for audio"""
    print("=" * 70)
    print("LIVE OCR DEMONSTRATION - Text Chunking for Audio")
    print("=" * 70)
    print()
    
    long_text = "The quick brown fox jumps over the lazy dog while the cat watches from the window"
    
    print(f"Original text:\n  '{long_text}'")
    print(f"  ({len(long_text.split())} words)\n")
    
    chunk_sizes = [3, 5, 8, 10]
    
    for size in chunk_sizes:
        chunks = chunk_text_for_audio(long_text, chunk_size=size)
        print(f"Chunked (max {size} words per chunk):")
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}: '{chunk}' ({len(chunk.split())} words)")
        print()
    
    print("-" * 70)
    print()


def demo_text_formatting():
    """Demonstrate text formatting for speech"""
    print("=" * 70)
    print("LIVE OCR DEMONSTRATION - Text Formatting for Speech")
    print("=" * 70)
    print()
    
    test_cases = [
        "Multiple    spaces    between    words",
        "Text  with   irregular     spacing",
        "A B C D E F G H I J K",
        "   Leading and trailing spaces   ",
        "Normal text with proper spacing",
    ]
    
    for text in test_cases:
        formatted = format_text_for_speech(text)
        print(f"Input:  '{text}'")
        print(f"Output: '{formatted}'")
        print()
    
    print("-" * 70)
    print()


def demo_similarity_detection():
    """Demonstrate text similarity calculation"""
    print("=" * 70)
    print("LIVE OCR DEMONSTRATION - Text Similarity Detection")
    print("=" * 70)
    print()
    
    base_text = "Milk 2% Kirkland Organic"
    
    test_cases = [
        ("Milk 2% Kirkland Organic", "Exact match"),
        ("Milk 2% Kirkland", "Slightly different (missing word)"),
        ("Milk 2 percent Kirkland Organic", "Same meaning, different words"),
        ("Kirkland Organic Milk 2%", "Same words, different order"),
        ("Orange Juice 100% Tropicana", "Completely different"),
        ("", "Empty string"),
    ]
    
    print(f"Base text: '{base_text}'\n")
    
    for text, description in test_cases:
        similarity = calculate_text_similarity(base_text, text)
        is_similar = similarity >= 0.8
        
        print(f"Test: {description}")
        print(f"  Text: '{text}'")
        print(f"  Similarity: {similarity:.2f} ({'✓ SIMILAR' if is_similar else '✗ DIFFERENT'})")
        print()
    
    print("-" * 70)
    print()


def main():
    """Run all demonstrations"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                   LIVE OCR TOOL DEMONSTRATION                      ║")
    print("║                                                                    ║")
    print("║  This demo shows how the live OCR tool works without requiring    ║")
    print("║  actual OCR API. It demonstrates the core features:               ║")
    print("║                                                                    ║")
    print("║  • Text tracking to avoid repetition                              ║")
    print("║  • Smart chunking for audio delivery                              ║")
    print("║  • Text similarity detection                                      ║")
    print("║  • Speech formatting                                              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n")
    
    demo_text_tracking()
    input("Press ENTER to continue to chunking demo...")
    print("\n")
    
    demo_chunking()
    input("Press ENTER to continue to formatting demo...")
    print("\n")
    
    demo_text_formatting()
    input("Press ENTER to continue to similarity demo...")
    print("\n")
    
    demo_similarity_detection()
    
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  ✓ Text tracking prevents repetitive reading")
    print("  ✓ Chunking makes long text more digestible")
    print("  ✓ Formatting improves speech quality")
    print("  ✓ Similarity detection is smart and configurable")
    print()
    print("The live OCR tool is ready for use with Google Cloud Vision API!")
    print()


if __name__ == "__main__":
    main()
