import stanza

def test_stanza_ner():
    """Test NER using Stanford's Stanza model."""
    # Download and load English model
    stanza.download('en')
    nlp = stanza.Pipeline('en', processors='tokenize,ner')
    
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
    
    print("Testing Stanza NER with travel entities...\n")
    
    for text in test_texts:
        print(f"\nText: {text}")
        doc = nlp(text)
        
        print("Entities found:")
        for sent in doc.sentences:
            for ent in sent.ents:
                print(f"  {ent.text:<20} {ent.type:<10}")

if __name__ == "__main__":
    test_stanza_ner()
