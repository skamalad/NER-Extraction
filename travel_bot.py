from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import spacy
from datetime import datetime
import re
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TravelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TravelBot:
    def __init__(self):
        # Initialize better NER model
        self.ner = pipeline("ner", 
                          model="jean-baptiste/roberta-large-ner-english",
                          aggregation_strategy="simple")
        
        # Initialize better intent classification
        self.intent_classifier = pipeline("zero-shot-classification",
                                        model="facebook/bart-large-mnli")
        
        # Load better spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Enhanced intent list with examples
        self.intent_examples = {
            "book_flight": [
                "I want to book a flight",
                "Need plane tickets",
                "Looking for air travel",
                "Book an airplane ticket",
                "Find me flight options"
            ],
            "hotel_search": [
                "Find me a hotel",
                "Need accommodation",
                "Looking for a place to stay",
                "Book a room",
                "Hotel reservation"
            ],
            "restaurant_search": [
                "Find restaurants",
                "Places to eat",
                "Food options",
                "Dining places",
                "Restaurant recommendations"
            ],
            "car_rental": [
                "Rent a car",
                "Need a vehicle",
                "Car hire",
                "Automobile rental",
                "Looking for a rental car"
            ],
            "weather_inquiry": [
                "What's the weather",
                "Temperature forecast",
                "Will it rain",
                "Weather conditions",
                "Climate information"
            ],
            "tourist_attraction": [
                "Places to visit",
                "Tourist spots",
                "Attractions nearby",
                "Sightseeing options",
                "Popular destinations"
            ]
        }

        # Custom entity patterns for travel domain
        self.custom_patterns = [
            {"label": "TRANSPORT", "pattern": [{"LOWER": {"IN": ["flight", "plane", "train", "bus", "taxi", "car"]}}]},
            {"label": "ACCOMMODATION", "pattern": [{"LOWER": {"IN": ["hotel", "hostel", "resort", "apartment", "room"]}}]},
            {"label": "AMENITY", "pattern": [{"LOWER": {"IN": ["wifi", "pool", "gym", "breakfast", "parking"]}}]},
            {"label": "PRICE_RANGE", "pattern": [{"LOWER": {"IN": ["cheap", "expensive", "budget", "luxury", "affordable"]}}]}
        ]
        
        # Add custom patterns to spaCy
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(self.custom_patterns)

    def extract_dates(self, text):
        """Enhanced date extraction with travel context"""
        doc = self.nlp(text)
        dates = []
        date_patterns = [
            r'\d{1,2}\/\d{1,2}\/\d{2,4}',  # DD/MM/YYYY
            r'\d{1,2}\-\d{1,2}\-\d{2,4}',   # DD-MM-YYYY
            r'next \w+',                     # next week/month
            r'this \w+',                     # this weekend
            r'\d{1,2}(st|nd|rd|th)?(\s+of)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(\s+\d{4})?'  # 25th of December
        ]
        
        # Extract dates from spaCy entities
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                dates.append({"text": ent.text, "type": ent.label_, "confidence": 0.9})
        
        # Extract dates from patterns
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if not any(d["text"] == match.group() for d in dates):
                    dates.append({"text": match.group(), "type": "DATE", "confidence": 0.8})
        
        return dates

    def classify_intent(self, text):
        """Enhanced intent classification with examples"""
        # Prepare candidate labels with examples
        candidate_labels = list(self.intent_examples.keys())
        
        # Get initial classification
        result = self.intent_classifier(
            text, 
            candidate_labels,
            hypothesis_template="This text is about {}."
        )
        
        # Enhance confidence by comparing with examples
        enhanced_scores = []
        for intent, examples in self.intent_examples.items():
            example_scores = []
            for example in examples:
                similarity = self.calculate_similarity(text.lower(), example.lower())
                example_scores.append(similarity)
            enhanced_scores.append(max(example_scores))
        
        # Combine original and enhanced scores
        final_scores = []
        for orig_score, enh_score in zip(result["scores"], enhanced_scores):
            final_scores.append((orig_score + enh_score) / 2)
        
        # Normalize scores
        total = sum(final_scores)
        final_scores = [score/total for score in final_scores]
        
        return {
            "intent": result["labels"][np.argmax(final_scores)],
            "confidence": max(final_scores)
        }

    def calculate_similarity(self, text1, text2):
        """Calculate text similarity using spaCy"""
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)

    def extract_entities(self, text):
        """Enhanced entity extraction with custom rules"""
        # Get entities from transformer model
        entities = self.ner(text)
        
        # Get entities from spaCy with custom rules
        doc = self.nlp(text)
        
        # Combine entities
        grouped = {}
        
        # Add transformer entities
        for ent in entities:
            ent_type = ent["entity_group"]
            if ent_type not in grouped:
                grouped[ent_type] = []
            grouped[ent_type].append({
                "text": ent["word"],
                "confidence": ent["score"]
            })
        
        # Add spaCy entities
        for ent in doc.ents:
            ent_type = ent.label_
            if ent_type not in grouped:
                grouped[ent_type] = []
            grouped[ent_type].append({
                "text": ent.text,
                "confidence": 0.8  # Default confidence for rule-based entities
            })
        
        return grouped

    def process_message(self, text):
        """Process a message with enhanced confidence scores"""
        # Get intent
        intent = self.classify_intent(text)
        
        # Get entities
        entities = self.extract_entities(text)
        
        # Get dates
        dates = self.extract_dates(text)
        
        # Calculate overall confidence
        entity_confidences = []
        for entity_list in entities.values():
            for entity in entity_list:
                entity_confidences.append(entity["confidence"])
        
        date_confidences = [date["confidence"] for date in dates]
        
        # Average confidence across all components
        confidences = [intent["confidence"]] + entity_confidences + date_confidences
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "text": text,
            "intent": intent,
            "entities": entities,
            "dates": dates,
            "overall_confidence": overall_confidence
        }

def main():
    # Initialize bot
    bot = TravelBot()
    
    # Test messages with more complex scenarios
    test_messages = [
        "I need a business class flight from New York to London next Friday, preferably in the morning",
        "Find me a 5-star hotel near the Eiffel Tower in Paris with a pool and free breakfast for December 25th",
        "What are the best Japanese restaurants in Tokyo that serve vegetarian sushi and have outdoor seating?",
        "I want to rent a luxury SUV in San Francisco from March 15th to March 18th",
        "What's the weather forecast for Dubai during the first week of April?",
        "Book me a table for two at a romantic Italian restaurant near the Spanish Steps in Rome for tomorrow evening",
        "I need a shuttle service from JFK airport to Manhattan on Tuesday at 2pm",
        "Are there any guided tours of the Great Wall of China available next month?"
    ]
    
    # Process each message
    for message in test_messages:
        print("\nProcessing:", message)
        print("-" * 80)
        
        result = bot.process_message(message)
        
        print(f"Intent: {result['intent']['intent']} (Confidence: {result['intent']['confidence']:.2%})")
        
        print("\nEntities:")
        for entity_type, entities in result["entities"].items():
            for entity in entities:
                print(f"{entity_type}: {entity['text']} (Confidence: {entity['confidence']:.2%})")
        
        print("\nDates:")
        for date in result["dates"]:
            print(f"{date['type']}: {date['text']} (Confidence: {date['confidence']:.2%})")
        
        print(f"\nOverall Confidence: {result['overall_confidence']:.2%}")
        print("-" * 80)

if __name__ == "__main__":
    main()
