import spacy
from spacy import displacy

def test_spacy_ner():
    """Test NER using Spacy's pre-trained model."""
    # Load English model
    nlp = spacy.load("en_core_web_trf")
    
    # Add travel-specific rules
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        # Landmarks
        {"label": "LANDMARK", "pattern": "Gateway of India"},
        {"label": "LANDMARK", "pattern": "Taj Mahal"},
        {"label": "LANDMARK", "pattern": "Red Fort"},
        
        # Events
        {"label": "EVENT", "pattern": "Kumbh Mela"},
        {"label": "EVENT", "pattern": "Goa Carnival"},
        {"label": "EVENT", "pattern": "Diwali"},
        
        # Organizations
        {"label": "ORG", "pattern": "SpiceJet"},
        {"label": "ORG", "pattern": "Air India"},
        {"label": "ORG", "pattern": "Taj Hotels"}
    ]
    ruler.add_patterns(patterns)
    
    # Test sentences
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
    
    print("Testing Spacy NER with travel-specific entities...\n")
    
    for text in test_texts:
        print(f"\nText: {text}")
        doc = nlp(text)
        
        print("Entities found:")
        for ent in doc.ents:
            print(f"  {ent.text:<20} {ent.label_:<10} {spacy.explain(ent.label_)}")

if __name__ == "__main__":
    test_spacy_ner()
