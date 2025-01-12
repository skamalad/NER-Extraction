import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os

def create_travel_domain_weights():
    # Initialize tokenizer and model
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    # Define travel-specific tags
    tags = ['O',                  # Outside of named entity
            'B-LOC', 'I-LOC',     # Location
            'B-ORG', 'I-ORG',     # Organization
            'B-TRANSPORT', 'I-TRANSPORT',  # Transportation
            'B-ACCOMMODATION', 'I-ACCOMMODATION']  # Accommodation
    
    num_states = len(tags)
    vocab_size = tokenizer.vocab_size
    
    # Create transition matrix (log probabilities)
    transitions = np.full((num_states, num_states), -10.0, dtype=np.float32)
    
    # Set valid transitions
    for i in range(num_states):
        for j in range(num_states):
            if i == 0:  # From O tag
                if j == 0:  # O -> O
                    transitions[i,j] = 0  # Common case
                elif j % 2 == 1:  # O -> B-*
                    transitions[i,j] = -1  # Slightly less common
            elif i % 2 == 1:  # From B-* tag
                if j == i + 1:  # B-X -> I-X
                    transitions[i,j] = 0  # Common case
                elif j == 0:    # B-X -> O
                    transitions[i,j] = -1  # Less common
            else:  # From I-* tag
                if j == i:      # I-X -> I-X
                    transitions[i,j] = 0  # Common case
                elif j == 0:    # I-X -> O
                    transitions[i,j] = -1  # Less common
                elif j % 2 == 1:  # I-X -> B-Y
                    transitions[i,j] = -2  # Rare but possible
    
    # Create emission matrix
    emissions = np.full((vocab_size, num_states), -10.0, dtype=np.float32)
    
    # Define travel-specific vocabulary with entity types
    travel_vocab = {
        'LOC': [
            "New York", "London", "Paris", "Tokyo", "Dubai", "Manhattan", 
            "airport", "beach", "station", "downtown", "city", "street", 
            "avenue", "terminal", "mountain", "lake", "river", "island",
            "coast", "bay", "harbor", "port", "district", "region",
            "continent", "country", "state", "province", "town", "village",
            "Eiffel Tower", "Mount Fuji", "Napa Valley", "JFK", "LAX"
        ],
        'ORG': [
            "Hilton", "Marriott", "Hyatt", "Sheraton", "Westin",
            "Airlines", "Airways", "Air", "Railway", "Metro",
            "Hotel", "Resort", "Restaurant", "Cafe", "Airport",
            "Station", "Agency", "Tours", "Travel", "Tourism",
            "French Laundry", "British Airways", "United", "Delta"
        ],
        'TRANSPORT': [
            "flight", "plane", "train", "bus", "taxi", "car",
            "shuttle", "metro", "subway", "ferry", "boat", "ship",
            "aircraft", "helicopter", "tram", "rail", "railway",
            "transit", "transport", "transportation", "ride", "rental",
            "Uber", "Lyft", "bullet train", "high-speed rail"
        ],
        'ACCOMMODATION': [
            "hotel", "hostel", "resort", "apartment", "room",
            "suite", "villa", "cottage", "inn", "motel",
            "lodging", "accommodation", "residence", "stay",
            "booking", "reservation", "check-in", "check-out",
            "five-star", "luxury", "boutique", "penthouse"
        ]
    }
    
    # Initialize special tokens
    special_tokens = {
        'O': ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '.', ',', '!', '?', 
              'the', 'a', 'an', 'in', 'on', 'at', 'to', 'from', 'with',
              'and', 'or', 'but', 'for', 'of', 'by', 'near', 'next',
              'want', 'need', 'looking', 'search', 'find', 'book', 'reserve']
    }
    
    # Process special tokens
    for tag, tokens in special_tokens.items():
        tag_idx = tags.index(tag)
        for token in tokens:
            if token in tokenizer.vocab:
                token_id = tokenizer.convert_tokens_to_ids(token)
                emissions[token_id, tag_idx] = 0  # High prob for O tag
                emissions[token_id, 1:] = -100  # Very low prob for entity tags
    
    # Process each entity type
    for entity_type, words in travel_vocab.items():
        # Get corresponding tag indices
        if entity_type == 'LOC':
            b_idx, i_idx = tags.index('B-LOC'), tags.index('I-LOC')
        elif entity_type == 'ORG':
            b_idx, i_idx = tags.index('B-ORG'), tags.index('I-ORG')
        elif entity_type == 'TRANSPORT':
            b_idx, i_idx = tags.index('B-TRANSPORT'), tags.index('I-TRANSPORT')
        elif entity_type == 'ACCOMMODATION':
            b_idx, i_idx = tags.index('B-ACCOMMODATION'), tags.index('I-ACCOMMODATION')
        
        # Process each word
        for word in words:
            # Handle multi-token words
            tokens = tokenizer.tokenize(word)
            for i, token in enumerate(tokens):
                token_id = tokenizer.convert_tokens_to_ids(token)
                if i == 0:  # First token gets B- tag
                    emissions[token_id, b_idx] = 0  # High prob for B- tag
                    emissions[token_id, 0] = -2  # Lower prob for O tag
                    # Very low prob for other entity tags
                    for j in range(1, num_states):
                        if j != b_idx:
                            emissions[token_id, j] = -10
                else:  # Subsequent tokens get I- tag
                    emissions[token_id, i_idx] = 0  # High prob for I- tag
                    emissions[token_id, 0] = -2  # Lower prob for O tag
                    # Very low prob for other entity tags
                    for j in range(1, num_states):
                        if j != i_idx:
                            emissions[token_id, j] = -10
    
    # Handle subword tokens (##)
    for token_id in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        if token.startswith('##'):
            # For subwords, encourage continuing the previous tag
            emissions[token_id] = -2  # Moderate probability for all tags
            emissions[token_id, 0] = -5  # Lower probability for O tag
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save weights
    np.savez('model/weights.npz', 
             transitions=transitions,
             emissions=emissions)
    
    print(f"Created weights with shapes: transitions={transitions.shape}, emissions={emissions.shape}")
    print(f"Data ranges: transitions=[{transitions.min():.3f}, {transitions.max():.3f}], "
          f"emissions=[{emissions.min():.3f}, {emissions.max():.3f}]")

if __name__ == "__main__":
    create_travel_domain_weights()