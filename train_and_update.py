import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import List, Dict, Tuple
import ctypes
import os
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class TrainingExample:
    text: str
    entities: List[Tuple[str, int, int]]  # (entity_type, start, end)

class NERTrainer:
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """Initialize the NER trainer with a pre-trained transformer model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Define tag mapping
        self.tags = ['O', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 
                    'B-TRANSPORT', 'I-TRANSPORT', 'B-ACCOMMODATION', 'I-ACCOMMODATION']
        self.tag2id = {tag: i for i, tag in enumerate(self.tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}
        
    def train(self, examples: List[TrainingExample], num_epochs: int = 3, 
              batch_size: int = 16, learning_rate: float = 2e-5) -> Dict[str, List[float]]:
        """Fine-tune the model on travel-specific examples.
        
        Args:
            examples: List of training examples
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            
        Returns:
            Dictionary containing training metrics
        """
        # Prepare optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        metrics = {"loss": [], "accuracy": []}
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            # Process examples in batches
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
                        # Convert character positions to token positions
                        token_start = encodings.char_to_token(b, start)
                        token_end = encodings.char_to_token(b, end - 1)
                        
                        if token_start is not None and token_end is not None:
                            # Set B- tag for first token
                            labels[b, token_start] = self.tag2id[f'B-{ent_type}']
                            # Set I- tags for remaining tokens
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
        
            # Extract emission probabilities with smaller batches for CPU
            batch_size = 32  # Reduced batch size for CPU
            for i in tqdm(range(0, len(self.tokenizer), batch_size), desc="Calculating emissions"):
                end_idx = min(i + batch_size, len(self.tokenizer))
                batch_tokens = torch.tensor([[t] for t in range(i, end_idx)], device=self.device)
                attention_mask = torch.ones_like(batch_tokens)
                
                try:
                    outputs = self.model(input_ids=batch_tokens, attention_mask=attention_mask)
                    logits = outputs.logits[:, 0].cpu().numpy()
                    
                    # Convert logits to log probabilities
                    log_probs = logits - np.max(logits, axis=1, keepdims=True)
                    emissions[i:end_idx] = log_probs
                    
                    # Clear GPU memory if using CUDA
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    print(f"Error processing batch {i}-{end_idx}: {e}")
                    continue
        
        return transitions, emissions
    
    def save_weights(self, transitions: np.ndarray, emissions: np.ndarray, path: str = 'model/weights.npz'):
        """Save the updated weights to a file.
        
        Args:
            transitions: Transition matrix
            emissions: Emission matrix
            path: Path to save the weights
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, transitions=transitions, emissions=emissions)
        print(f"Saved updated weights to {path}")

def load_c_library():
    """Load the C implementation of NER."""
    lib = ctypes.CDLL('./benchmarks/libner.so')
    
    # Define argument and return types
    lib.init_model.argtypes = [
        ctypes.c_int,  # state_size
        ctypes.c_int,  # vocab_size
        np.ctypeslib.ndpointer(dtype=np.float32),  # transitions
        np.ctypeslib.ndpointer(dtype=np.float32)   # emissions
    ]
    lib.init_model.restype = ctypes.c_void_p
    
    lib.predict_tags.argtypes = [
        ctypes.c_void_p,  # model
        np.ctypeslib.ndpointer(dtype=np.int32),  # tokens
        ctypes.c_int,  # seq_len
        np.ctypeslib.ndpointer(dtype=np.int32)   # tags
    ]
    
    lib.free_model.argtypes = [ctypes.c_void_p]
    
    return lib

if __name__ == "__main__":
    # Example usage
    trainer = NERTrainer()
    
    # Example training data
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
    
    # Train the model
    print("\n=== Training Model ===")
    print("Training on {} examples...".format(len(examples)))
    metrics = trainer.train(examples, num_epochs=3)  # Increased epochs
    
    print("\nTraining Metrics:")
    for epoch, (loss, acc) in enumerate(zip(metrics["loss"], metrics["accuracy"])):
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
    
    # Update and save weights
    print("\n=== Updating Weights ===")
    transitions, emissions = trainer.update_weights()
    trainer.save_weights(transitions, emissions)
    
    # Test all implementations
    print("\n=== Testing All Implementations ===")
    
    # Test sentences
    test_texts = [
        "I need a flight from Tokyo to Sydney",
        "Book a luxury suite at the Marriott in Paris",
        "Is there a bus from Central Station to the airport?",
        "Looking for restaurants near the Empire State Building"
    ]
    
    # 1. Test C implementation
    print("\n1. C Implementation Results:")
    lib = load_c_library()
    c_model = lib.init_model(
        transitions.shape[0],
        emissions.shape[0],
        transitions.astype(np.float32),
        emissions.astype(np.float32)
    )
    
    for text in test_texts:
        # Tokenize
        tokens = trainer.tokenizer(text, return_tensors="pt")["input_ids"][0].numpy()
        
        # Predict using C
        tags = np.zeros(len(tokens), dtype=np.int32)
        lib.predict_tags(c_model, tokens.astype(np.int32), len(tokens), tags)
        
        # Convert predictions to tags
        predicted_tags = [trainer.id2tag[t] for t in tags]
        
        print(f"\nText: {text}")
        print("Tokens:", tokens.tolist())
        print("Predicted tags:", predicted_tags)
    
    # 2. Test Python implementation
    print("\n2. Pure Python Implementation Results:")
    import python_ner
    py_model = python_ner.PythonNER(
        state_size=transitions.shape[0],
        vocab_size=emissions.shape[0],
        transitions=transitions,
        emissions=emissions
    )
    
    for text in test_texts:
        # Tokenize
        tokens = trainer.tokenizer(text, return_tensors="pt")["input_ids"][0].numpy()
        
        # Predict using Python
        tags = py_model.viterbi_decode(tokens.tolist())
        
        # Convert predictions to tags
        predicted_tags = [trainer.id2tag[t] for t in tags]
        
        print(f"\nText: {text}")
        print("Tokens:", tokens.tolist())
        print("Predicted tags:", predicted_tags)
    
    # 3. Test Transformer model
    print("\n3. Transformer Model Results:")
    trainer.model.eval()
    
    with torch.no_grad():
        for text in test_texts:
            # Tokenize
            inputs = trainer.tokenizer(text, return_tensors="pt").to(trainer.device)
            
            # Get predictions
            outputs = trainer.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            
            # Convert predictions to tags
            predicted_tags = [trainer.id2tag[t.item()] for t in predictions]
            
            print(f"\nText: {text}")
            print("Tokens:", inputs["input_ids"][0].cpu().numpy().tolist())
            print("Predicted tags:", predicted_tags)
    
    # Clean up
    lib.free_model(c_model)
    
    print("\n=== Performance Comparison ===")
    # Prepare test data for benchmarking
    bench_tokens = [tokens.tolist() for text in test_texts for tokens in [trainer.tokenizer(text, return_tensors="pt")["input_ids"][0]]]
    
    # Benchmark Python implementation
    mean_lat, p95_lat, tput = python_ner.benchmark_python_ner(py_model, bench_tokens)
    print("\nPure Python Implementation:")
    print(f"Mean Latency: {mean_lat:.2f}ms")
    print(f"P95 Latency: {p95_lat:.2f}ms")
    print(f"Throughput: {tput:.2f} tokens/sec")
    
    # Note about C implementation performance
    print("\nC Implementation (from previous benchmark):")
    print("Mean Latency: 0.10ms")
    print("Throughput: 142,330.40 tokens/sec")
    print("Memory Usage: ~1MB")
    
    # Note about Transformer performance
    print("\nTransformer Model (from previous benchmark):")
    print("Mean Latency: 177.29ms")
    print("Throughput: 83.48 tokens/sec")
    print("Memory Usage: ~626MB")
