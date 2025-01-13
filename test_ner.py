import sys
import os
import numpy as np
from src.fast_ner import FastNER
from src.vocabulary import Vocabulary

def test_sentences():
    """Test the NER model with various sentences."""
    test_texts = [
        "I want to visit the Taj Mahal in Agra during Diwali",
        "Book a room at Taj Hotels near Gateway of India in Mumbai",
        "What's the best time to visit Manali and Shimla",
        "Looking for hotels near Charminar in Hyderabad",
        "Want to attend the Goa Carnival next month",
        "Take SpiceJet from Delhi to Chennai on Friday",
        "Show me tourist attractions near Mysore Palace",
        "How to reach Varanasi from Delhi during Kumbh Mela"
    ]
    
    print("Python: Starting test...")
    print(f"Python: Current directory: {os.getcwd()}")
    print(f"Python: Python version: {sys.version}")
    
    # Initialize model
    print("\nPython: Initializing FastNER...")
    model = FastNER("model/weights.npz")
    print("Model initialized")
    
    # Test each sentence
    for test_text in test_texts:
        print(f"\nPython: Test text: {test_text}")
        print("\nPython: Running inference...")
        
        # Tokenize and encode
        tokens = test_text.split()
        token_ids = model.vocab.encode(test_text, max_length=len(tokens))
        
        # Run inference
        tags, scores = model(token_ids)
        
        # Format results
        print("\nPython: Named Entities:")
        formatted_results = model.format_results(tokens, tags, scores)
        for token, tag, score in formatted_results:
            print(f"  {token:<12} {tag:<10} {score:>7.4f}")
    
    print("\nPython: Freeing model memory")
    model.free()

if __name__ == "__main__":
    import os
    import sys
    test_sentences()
