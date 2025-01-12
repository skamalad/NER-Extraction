# -*- coding: utf-8 -*-
"""
# Travel Domain NER with GPU Acceleration

This notebook implements a Named Entity Recognition (NER) system specifically for travel-related queries,
utilizing GPU acceleration for faster training and inference.

## Setup and Imports
"""

# Install required packages
# !pip install transformers tqdm numpy torch

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time
from tqdm import tqdm
import os
from google.colab import files

@dataclass
class TrainingExample:
    text: str
    entities: List[Tuple[str, int, int]]  # (entity_type, start, end)

"""## Pure Python NER Implementation"""

class PythonNER:
    def __init__(self, state_size: int, vocab_size: int, transitions: np.ndarray, emissions: np.ndarray):
        """Initialize the NER model with transition and emission matrices."""
        self.state_size = state_size
        self.vocab_size = vocab_size
        self.transitions = transitions
        self.emissions = emissions
        
    def viterbi_decode(self, tokens: List[int]) -> List[int]:
        """Decode a sequence of tokens into their most likely tags using Viterbi algorithm."""
        seq_len = len(tokens)
        if seq_len == 0:
            return []
            
        # Initialize Viterbi tables
        dp = np.full((seq_len, self.state_size), -np.inf)
        prev = np.zeros((seq_len, self.state_size), dtype=np.int32)
        
        # Initialize first position
        token_id = tokens[0]
        emit_probs = self.emissions[token_id]
        
        for s in range(self.state_size):
            if s == 0:  # O tag
                dp[0, s] = emit_probs[s]
            elif s % 2 == 1:  # B- tags
                dp[0, s] = emit_probs[s]
            else:  # I- tags
                dp[0, s] = -100.0  # Cannot start with I- tag
            prev[0, s] = -1
        
        # Forward pass
        for t in range(1, seq_len):
            token_id = tokens[t]
            emit_probs = self.emissions[token_id]
            is_subword = (token_id >= 1000 and token_id <= 2000)
            
            for curr_s in range(self.state_size):
                max_score = -np.inf
                best_prev = 0
                
                # For each possible previous state
                for prev_s in range(self.state_size):
                    trans_score = self.transitions[prev_s, curr_s]
                    score = dp[t-1, prev_s] + trans_score
                    
                    # Apply constraints based on tag type
                    if curr_s % 2 == 0 and curr_s > 0:  # I- tag
                        # Can only transition to I-X from B-X or I-X of same type
                        if prev_s != curr_s and prev_s != curr_s - 1:
                            score = -100.0
                            
                    if score > max_score:
                        max_score = score
                        best_prev = prev_s
                
                # Add emission score
                if is_subword and t > 0:
                    # For subwords, strongly prefer continuing the previous tag
                    prev_tag = np.argmax(dp[t-1])
                    if curr_s == prev_tag:
                        dp[t, curr_s] = max_score
                    else:
                        dp[t, curr_s] = -100.0
                else:
                    dp[t, curr_s] = max_score + emit_probs[curr_s]
                    
                prev[t, curr_s] = best_prev
        
        # Backward pass to recover best path
        tags = np.zeros(seq_len, dtype=np.int32)
        tags[-1] = np.argmax(dp[-1])
        
        for t in range(seq_len-2, -1, -1):
            tags[t] = prev[t+1, tags[t+1]]
            
        return tags.tolist()

"""## NER Trainer with GPU Support"""

class NERTrainer:
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """Initialize the NER trainer with a pre-trained transformer model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Define tag mapping
        self.tags = ['O', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 
                    'B-TRANSPORT', 'I-TRANSPORT', 'B-ACCOMMODATION', 'I-ACCOMMODATION']
        self.tag2id = {tag: i for i, tag in enumerate(self.tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}
        
    def train(self, examples: List[TrainingExample], num_epochs: int = 3, 
              batch_size: int = 16, learning_rate: float = 2e-5) -> Dict[str, List[float]]:
        """Fine-tune the model on travel-specific examples."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        metrics = {"loss": [], "accuracy": []}
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            for i in tqdm(range(0, len(examples), batch_size)):
                batch = examples[i:i + batch_size]
                
                # Prepare batch inputs
                texts = [ex.text for ex in batch]
                encodings = self.tokenizer(texts, padding=True, truncation=True, 
                                        return_tensors="pt").to(self.device)
                
                # Prepare labels
                labels = torch.full((len(batch), encodings.input_ids.shape[1]), 
                                 self.tag2id['O'], device=self.device)
                
                # Fill in entity labels
                for b, example in enumerate(batch):
                    for ent_type, start, end in example.entities:
                        token_start = encodings.char_to_token(b, start)
                        token_end = encodings.char_to_token(b, end - 1)
                        
                        if token_start is not None and token_end is not None:
                            labels[b, token_start] = self.tag2id[f'B-{ent_type}']
                            for t in range(token_start + 1, token_end + 1):
                                labels[b, t] = self.tag2id[f'I-{ent_type}']
                
                # Forward pass
                outputs = self.model(**encodings, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                predictions = torch.argmax(logits, dim=-1)
                mask = encodings.attention_mask.bool()
                correct = (predictions == labels)[mask].sum().item()
                total = mask.sum().item()
                
                epoch_loss += loss.item()
                epoch_correct += correct
                epoch_total += total
            
            # Compute epoch metrics
            avg_loss = epoch_loss / (len(examples) / batch_size)
            accuracy = epoch_correct / epoch_total
            metrics["loss"].append(avg_loss)
            metrics["accuracy"].append(accuracy)
            
            print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def update_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate updated transition and emission matrices from the fine-tuned model."""
        self.model.eval()
        
        num_tags = len(self.tags)
        transitions = np.full((num_tags, num_tags), -10.0, dtype=np.float32)
        emissions = np.full((len(self.tokenizer), num_tags), -10.0, dtype=np.float32)
        
        # Extract transition probabilities
        with torch.no_grad():
            if hasattr(self.model, 'crf'):
                trans_matrix = self.model.crf.transitions.cpu().numpy()
                transitions = trans_matrix
            else:
                # Use heuristic rules
                for i in range(num_tags):
                    for j in range(num_tags):
                        if i == 0:  # From O tag
                            if j == 0:  # O -> O
                                transitions[i,j] = 0
                            elif j % 2 == 1:  # O -> B-*
                                transitions[i,j] = -1
                        elif i % 2 == 1:  # From B-* tag
                            if j == i + 1:  # B-X -> I-X
                                transitions[i,j] = 0
                            elif j == 0:    # B-X -> O
                                transitions[i,j] = -1
                        else:  # From I-* tag
                            if j == i:      # I-X -> I-X
                                transitions[i,j] = 0
                            elif j == 0:    # I-X -> O
                                transitions[i,j] = -1
                            elif j % 2 == 1:  # I-X -> B-Y
                                transitions[i,j] = -2
        
            # Extract emission probabilities using GPU
            batch_size = 128
            for i in tqdm(range(0, len(self.tokenizer), batch_size)):
                batch_tokens = torch.arange(i, min(i + batch_size, len(self.tokenizer)), 
                                         device=self.device).unsqueeze(1)
                attention_mask = torch.ones_like(batch_tokens)
                
                outputs = self.model(input_ids=batch_tokens, attention_mask=attention_mask)
                logits = outputs.logits[:, 0].cpu().numpy()
                
                # Convert logits to log probabilities
                log_probs = logits - np.max(logits, axis=1, keepdims=True)
                emissions[i:i + len(log_probs)] = log_probs
        
        return transitions, emissions
    
    def save_weights(self, transitions: np.ndarray, emissions: np.ndarray, path: str = 'weights.npz'):
        """Save the updated weights to a file."""
        np.savez(path, transitions=transitions, emissions=emissions)
        print(f"Saved weights to {path}")
        
        # For Colab: Download the weights file
        files.download(path)

"""## Example Usage and Testing"""

def main():
    # Check GPU availability
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Create training examples
    examples = [
        TrainingExample(
            text="I want to book a flight from New York to London",
            entities=[
                ("TRANSPORT", 17, 23),  # flight
                ("LOC", 29, 37),        # New York
                ("LOC", 41, 47)         # London
            ]
        ),
        TrainingExample(
            text="Looking for a 5-star hotel near the Eiffel Tower",
            entities=[
                ("ACCOMMODATION", 19, 24),  # hotel
                ("LOC", 34, 46)            # Eiffel Tower
            ]
        ),
        TrainingExample(
            text="Need a taxi from JFK Airport to Manhattan",
            entities=[
                ("TRANSPORT", 7, 11),   # taxi
                ("LOC", 17, 27),        # JFK Airport
                ("LOC", 31, 39)         # Manhattan
            ]
        ),
        TrainingExample(
            text="Book a suite at the Hilton Resort in Dubai",
            entities=[
                ("ACCOMMODATION", 7, 12),  # suite
                ("ORG", 20, 32),          # Hilton Resort
                ("LOC", 36, 41)           # Dubai
            ]
        ),
        TrainingExample(
            text="Is there a direct train from Paris to Rome?",
            entities=[
                ("TRANSPORT", 16, 21),  # train
                ("LOC", 27, 32),        # Paris
                ("LOC", 36, 40)         # Rome
            ]
        )
    ]
    
    # Initialize and train model
    trainer = NERTrainer()
    print("\n=== Training Model ===")
    metrics = trainer.train(examples, num_epochs=5)  # Increased epochs for better results
    
    # Generate and save weights
    print("\n=== Generating Weights ===")
    transitions, emissions = trainer.update_weights()
    trainer.save_weights(transitions, emissions)
    
    # Test the model
    print("\n=== Testing Model ===")
    test_texts = [
        "I need a flight from Tokyo to Sydney",
        "Book a luxury suite at the Marriott in Paris",
        "Is there a bus from Central Station to the airport?",
        "Looking for restaurants near the Empire State Building"
    ]
    
    trainer.model.eval()
    with torch.no_grad():
        for text in test_texts:
            inputs = trainer.tokenizer(text, return_tensors="pt").to(trainer.device)
            outputs = trainer.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            predicted_tags = [trainer.id2tag[t.item()] for t in predictions]
            
            print(f"\nText: {text}")
            print("Predicted tags:", predicted_tags)
    
    print("\nWeights have been saved and can be downloaded for use with the C implementation.")

if __name__ == "__main__":
    main()
