import sys
from src.fast_ner import FastNER  # This uses the C implementation

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

def test_ner():
    print("=== Travel NER Test ===")
    print(f"Python version: {sys.version}")
    
    # Initialize NER
    ner = FastNER("model/weights.npz")
    
    # Test cases
    test_sentences = [
        "I want to fly from New York to Paris on Air France",
        "Book a room at the Hilton Hotel in London",
        "Take the bullet train from Tokyo to Kyoto",
        "I'm looking for restaurants near the Eiffel Tower",
        "Emirates Airlines flies from Dubai to Singapore daily",
        "The cruise ship departs from Miami to Caribbean islands"
    ]
    
    for sentence in test_sentences:
        test_phrase(ner, sentence)

if __name__ == "__main__":
    try:
        test_ner()
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
