import numpy as np
import json
from typing import List, Tuple, Dict
import random
from pathlib import Path
from . import india_travel_data as india_data

class TravelDataGenerator:
    def __init__(self):
        # Load Google Maps data if available
        self.maps_data = self._load_maps_data()
        
        # Define entity types using both Maps and static data
        self.entity_types = {
            'LOC': self._get_locations(),
            'ORG': self._get_organizations(),
            'LANDMARK': self._get_landmarks(),
            'EVENT': india_data.get_all_events(),
            'TIME': ['morning', 'afternoon', 'evening', 'night',
                    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                    'January', 'February', 'March', 'April', 'May', 'June',
                    'next week', 'next month', 'tomorrow', 'tonight']
        }
        
        print(f"Loaded entities:")
        for entity_type, entities in self.entity_types.items():
            print(f"  {entity_type}: {len(entities)} items")
        
        # Templates for generating sentences
        self.templates = [
            # Location-based templates
            "I want to fly from {LOC} to {LOC}",
            "Book a flight from {LOC} to {LOC} for {TIME}",
            "Looking for hotels in {LOC}",
            "Planning a trip to {LOC}",
            "What's the weather like in {LOC}",
            "How far is {LOC} from {LOC}",
            "Best time to visit {LOC}",
            "Show me places to visit in {LOC}",
            "Top rated attractions in {LOC}",
            
            # Landmark-based templates
            "Looking for hotels near {LANDMARK} in {LOC}",
            "Visit the {LANDMARK} when you're in {LOC}",
            "Take a tour of the {LANDMARK} on {TIME}",
            "How to reach {LANDMARK}",
            "Opening hours of {LANDMARK}",
            "Book tickets for {LANDMARK}",
            "Is {LANDMARK} open today",
            "Reviews of {LANDMARK}",
            
            # Organization-based templates
            "{ORG} operates flights between {LOC} and {LOC}",
            "Book a room at {ORG} in {LOC} for {TIME}",
            "{ORG} offers great deals for flights to {LOC}",
            "Reviews of {ORG} in {LOC}",
            "Contact {ORG} customer service",
            "Does {ORG} operate in {LOC}",
            
            # Event-based templates
            "Want to attend the {EVENT} in {LOC}",
            "Going to {EVENT} celebrations in {LOC}",
            "Book tickets for {EVENT} festival",
            "When is {EVENT} in {LOC}",
            "What to expect at {EVENT}",
            "Best places to celebrate {EVENT}",
            "How to reach {EVENT} venue",
            
            # Mixed templates
            "Visit {LOC} during {EVENT}",
            "Don't miss the {EVENT} when in {LOC}",
            "Stay at {ORG} near {LANDMARK}",
            "Take {ORG} to visit {LANDMARK}",
            "Experience {EVENT} at {LANDMARK}",
            "Best hotels near {LANDMARK}",
            "How to get from {LANDMARK} to {LANDMARK}",
            "Book {ORG} tour of {LANDMARK}"
        ]
    
    def _load_maps_data(self) -> Dict:
        """Load location data from Google Maps cache."""
        maps_data_path = Path('data/india_locations.json')
        if not maps_data_path.exists():
            print("Google Maps data not found. Using only static data.")
            return {}
        
        try:
            with open(maps_data_path, 'r') as f:
                data = json.load(f)
            print("Loaded Google Maps data successfully")
            return data
        except Exception as e:
            print(f"Error loading Google Maps data: {e}")
            return {}
    
    def _get_locations(self) -> List[str]:
        """Get locations from both Maps and static data."""
        locations = set(india_data.get_all_locations())
        
        # Add international locations
        locations.update([
            'New York', 'London', 'Paris', 'Tokyo', 'Dubai', 'Singapore',
            'Miami', 'San Francisco', 'Rome', 'Barcelona', 'Venice',
            'Hong Kong', 'Sydney', 'Bangkok', 'Istanbul', 'Cairo',
            'Las Vegas', 'Amsterdam', 'Berlin', 'Moscow'
        ])
        
        # Add locations from Maps data
        if self.maps_data and 'cities' in self.maps_data:
            for city_data in self.maps_data['cities'].values():
                if 'name' in city_data:
                    locations.add(city_data['name'])
        
        return list(locations)
    
    def _get_landmarks(self) -> List[str]:
        """Get landmarks from both Maps and static data."""
        landmarks = set(india_data.get_all_landmarks())
        
        # Add international landmarks
        landmarks.update([
            'Eiffel Tower', 'Big Ben', 'Statue of Liberty',
            'Golden Gate Bridge', 'Tower Bridge', 'Colosseum',
            'Great Wall', 'Sydney Opera House', 'Burj Khalifa',
            'Louvre Museum', 'Times Square'
        ])
        
        # Add landmarks from Maps data
        if self.maps_data:
            # Add attractions
            if 'attractions' in self.maps_data:
                for city_attractions in self.maps_data['attractions'].values():
                    for attraction in city_attractions:
                        if 'name' in attraction:
                            landmarks.add(attraction['name'])
            
            # Add temples
            if 'temples' in self.maps_data:
                for city_temples in self.maps_data['temples'].values():
                    for temple in city_temples:
                        if 'name' in temple:
                            landmarks.add(temple['name'])
            
            # Add other landmarks
            if 'landmarks' in self.maps_data:
                for city_landmarks in self.maps_data['landmarks'].values():
                    for landmark in city_landmarks:
                        if 'name' in landmark:
                            landmarks.add(landmark['name'])
        
        return list(landmarks)
    
    def _get_organizations(self) -> List[str]:
        """Get organizations from static data."""
        orgs = set(india_data.get_all_organizations())
        
        # Add international organizations
        orgs.update([
            'Air France', 'British Airways', 'Emirates', 'Delta',
            'United Airlines', 'Hilton', 'Marriott', 'Hyatt',
            'Four Seasons', 'Airbnb', 'Expedia', 'Booking.com',
            'TripAdvisor', 'Japan Airlines', 'Qatar Airways'
        ])
        
        return list(orgs)
    
    def generate_sentence(self) -> Tuple[str, List[Tuple[str, int, int]]]:
        """Generate a sentence and its entity annotations."""
        # Pick a random template
        template = random.choice(self.templates)
        
        # Track entities for annotation
        entities = []
        result = template
        
        # Replace each placeholder with a random entity
        for entity_type in self.entity_types:
            while f"{{{entity_type}}}" in result:
                entity = random.choice(self.entity_types[entity_type])
                placeholder = f"{{{entity_type}}}"
                
                # Find position of placeholder
                start_idx = result.find(placeholder)
                
                # Replace placeholder with entity
                result = result.replace(placeholder, entity, 1)
                
                # Record entity position
                entities.append((entity_type, start_idx, start_idx + len(entity)))
        
        return result, sorted(entities, key=lambda x: x[1])
    
    def generate_dataset(self, size: int) -> List[Tuple[str, List[Tuple[str, int, int]]]]:
        """Generate a dataset of the specified size."""
        return [self.generate_sentence() for _ in range(size)]
    
    def convert_to_bio_tags(self, text: str, entities: List[Tuple[str, int, int]]) -> List[str]:
        """Convert entity annotations to BIO tags."""
        # Initialize all tokens as O (Outside)
        words = text.split()
        tags = ['O'] * len(words)
        
        # Build token spans
        token_spans = []
        current_pos = 0
        for word in words:
            # Skip leading spaces
            while current_pos < len(text) and text[current_pos].isspace():
                current_pos += 1
            
            # Find word boundaries
            word_start = current_pos
            word_end = word_start + len(word)
            token_spans.append((word_start, word_end))
            current_pos = word_end
        
        # Assign B- and I- tags
        for entity_type, entity_start, entity_end in entities:
            # Find tokens that overlap with the entity
            entity_tokens = []
            for i, (token_start, token_end) in enumerate(token_spans):
                if (token_start <= entity_start < token_end or  # Token starts the entity
                    token_start < entity_end <= token_end or    # Token ends the entity
                    entity_start <= token_start < entity_end):  # Token is inside entity
                    entity_tokens.append(i)
            
            # Assign tags
            for i, token_idx in enumerate(entity_tokens):
                if i == 0:
                    tags[token_idx] = f'B-{entity_type}'
                else:
                    tags[token_idx] = f'I-{entity_type}'
        
        return tags

def create_training_data(num_samples: int = 1000) -> Tuple[List[str], List[List[str]]]:
    """Create training data with sentences and their BIO tags."""
    generator = TravelDataGenerator()
    sentences = []
    all_tags = []
    
    for _ in range(num_samples):
        text, entities = generator.generate_sentence()
        tags = generator.convert_to_bio_tags(text, entities)
        sentences.append(text)
        all_tags.append(tags)
    
    return sentences, all_tags
