import sys
import os
from src.fast_ner import FastNER

def test_phrase(ner, text):
    print(f"\n=== Testing: {text} ===")
    tokens = text.split()
    token_ids = ner.preprocess_text(text)
    tags, scores = ner(token_ids)
    
    # Format results
    results = ner.format_results(tokens, tags, scores)
    print("\nNamed Entities:")
    for token, tag, score in results:
        print(f"  {token:12} {tag:8} {score:8.4f}")

def main():
    try:
        print("=== Travel NER Test ===")
        print(f"Python version: {sys.version}")
        
        # Initialize NER
        ner = FastNER("model/weights.npz")
        
        # Test various travel-related phrases
        test_phrases = [
            "I want to fly from London to Tokyo next week",
            "Book a hotel in San Francisco near Golden Gate Bridge",
            "Take the train from Paris to Rome via Switzerland",
            "Emirates Airlines flies from Dubai to Singapore daily",
            "Looking for restaurants near Times Square in New York",
            "The cruise ship departs from Miami to Caribbean islands"
        ]
        
        for phrase in test_phrases:
            test_phrase(ner, phrase)
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
