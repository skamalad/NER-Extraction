import time
import numpy as np
import ctypes
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer

@dataclass
class BenchmarkResults:
    mean_latency: float
    std_latency: float
    p95_latency: float
    memory_used: float
    throughput: float

class FastNERBenchmark:
    def __init__(self, model_path: str, tokenizer_name: str = "bert-base-cased"):
        # Initialize C library
        self.lib_path = Path(__file__).parent / "libner.so"
        print(f"Loading library from: {self.lib_path}")
        self.lib = ctypes.CDLL(str(self.lib_path))
        self._setup_c_bindings()
        
        # Load model weights and initialize
        print(f"Loading weights from: {model_path}")
        weights = np.load(model_path)
        self.transitions = np.ascontiguousarray(weights['transitions'].astype(np.float32))
        self.emissions = np.ascontiguousarray(weights['emissions'].astype(np.float32))
        self.state_size = self.transitions.shape[0]
        self.vocab_size = self.emissions.shape[0]
        
        print(f"Initializing model with shapes: transitions={self.transitions.shape}, emissions={self.emissions.shape}")
        print(f"Data ranges: transitions=[{self.transitions.min():.3f}, {self.transitions.max():.3f}], emissions=[{self.emissions.min():.3f}, {self.emissions.max():.3f}]")
        
        # Initialize model
        self.model = self.lib.init_model(
            ctypes.c_int(self.state_size),
            ctypes.c_int(self.vocab_size),
            self.transitions,
            self.emissions
        )
        
        print(f"Model pointer: {self.model}")
        if not self.model:
            raise RuntimeError("Failed to initialize NER model")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def _setup_c_bindings(self):
        """Configure C function signatures"""
        # Load the shared library
        print(f"Loading library from: {self.lib_path}")
        self.lib = ctypes.CDLL(self.lib_path)
        
        # Set up function signatures for init_model
        self.lib.init_model.argtypes = [
            ctypes.c_int,  # state_size
            ctypes.c_int,  # vocab_size
            np.ctypeslib.ndpointer(dtype=np.float32),  # transitions
            np.ctypeslib.ndpointer(dtype=np.float32)   # emissions
        ]
        self.lib.init_model.restype = ctypes.c_void_p
        
        # Set up function signatures for predict_tags
        self.lib.predict_tags.argtypes = [
            ctypes.c_void_p,  # model
            np.ctypeslib.ndpointer(dtype=np.int32),  # tokens
            ctypes.c_int,  # seq_len
            np.ctypeslib.ndpointer(dtype=np.int32)   # tags
        ]
        self.lib.predict_tags.restype = None
        
        # Set up function signatures for free_model
        self.lib.free_model.argtypes = [ctypes.c_void_p]
        self.lib.free_model.restype = None

    def preprocess_text(self, text: str) -> np.ndarray:
        """Convert text to token IDs"""
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        return tokens['input_ids'][0]

    def run_inference(self, tokens):
        # Convert tokens to numpy array
        tokens_array = np.array(tokens, dtype=np.int32)
        seq_len = len(tokens)
        
        # Allocate output arrays
        tags = np.zeros(seq_len, dtype=np.int32)
        
        # Run inference
        self.lib.predict_tags(self.model, tokens_array, seq_len, tags)
        
        return tags, None  # Return None for scores since we don't compute them anymore

    def measure_latency(self, 
                       texts: List[str], 
                       num_runs: int = 100,
                       warmup_runs: int = 10) -> BenchmarkResults:
        """
        Measure inference latency including preprocessing
        """
        # Preprocessing latencies
        preprocess_times = []
        inference_times = []
        total_tokens = 0
        
        # Warmup runs
        warmup_tokens = [self.preprocess_text(text) for text in texts[:warmup_runs]]
        for tokens in warmup_tokens:
            _ = self.run_inference(tokens)
        
        # Benchmark runs
        start_memory = self.get_memory_usage()
        
        for text in texts:
            # Measure preprocessing time
            t0 = time.perf_counter()
            tokens = self.preprocess_text(text)
            t1 = time.perf_counter()
            preprocess_times.append((t1 - t0) * 1000)  # ms
            
            total_tokens += len(tokens)
            
            # Measure inference time
            t0 = time.perf_counter()
            _ = self.run_inference(tokens)
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) * 1000)  # ms
        
        end_memory = self.get_memory_usage()
        
        # Calculate statistics
        total_times = np.array(preprocess_times) + np.array(inference_times)
        total_time = np.sum(total_times)
        
        return BenchmarkResults(
            mean_latency=float(np.mean(total_times)),
            std_latency=float(np.std(total_times)),
            p95_latency=float(np.percentile(total_times, 95)),
            memory_used=end_memory - start_memory,
            throughput=total_tokens / (total_time / 1000)  # tokens/second
        )

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024**2

    def benchmark_comparison(self, 
                           texts: List[str],
                           compare_model_name: str = "dslim/bert-base-NER"):  
        """
        Compare performance with a standard transformer model
        """
        # Benchmark our C implementation
        c_results = self.measure_latency(texts)
        
        # Benchmark transformer model
        from transformers import AutoModelForTokenClassification
        
        model = AutoModelForTokenClassification.from_pretrained(compare_model_name)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        
        transformer_times = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                t0 = time.perf_counter()
                _ = model(**inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                transformer_times.append((t1 - t0) * 1000)
        
        return {
            "c_implementation": {
                "mean_latency": c_results.mean_latency,
                "p95_latency": c_results.p95_latency,
                "throughput": c_results.throughput,
                "memory": c_results.memory_used
            },
            "transformer": {
                "mean_latency": float(np.mean(transformer_times)),
                "p95_latency": float(np.percentile(transformer_times, 95)),
                "throughput": sum(len(self.tokenizer.encode(t)) for t in texts) / (sum(transformer_times) / 1000),
                "memory": self.get_memory_usage()
            }
        }

    def display_predictions(self, text: str):
        """Display NER predictions for a given text"""
        # Get tokens and run inference
        tokens = self.preprocess_text(text)
        tags, _ = self.run_inference(tokens)
        
        # Convert numeric tags back to labels
        tag_names = ['O', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-TRANSPORT', 'I-TRANSPORT', 
                    'B-ACCOMMODATION', 'I-ACCOMMODATION']
        
        # Get the original tokens
        token_texts = self.tokenizer.convert_ids_to_tokens(tokens)
        
        # Group entities
        current_entity = None
        current_text = []
        entities = []
        
        for token, tag_id in zip(token_texts, tags):
            tag = tag_names[tag_id]
            if tag == 'O':
                if current_entity:
                    entities.append((current_entity, ' '.join(current_text)))
                    current_entity = None
                    current_text = []
            elif tag.startswith('B-'):
                if current_entity:
                    entities.append((current_entity, ' '.join(current_text)))
                current_entity = tag[2:]
                current_text = [token]
            elif tag.startswith('I-'):
                if current_entity == tag[2:]:
                    current_text.append(token)
        
        if current_entity:
            entities.append((current_entity, ' '.join(current_text)))
        
        # Print results
        print(f"\nText: {text}")
        print("Entities found:")
        for entity_type, text in entities:
            print(f"- {entity_type}: {text}")
        print()

    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, 'model'):
            self.lib.free_model(self.model)

if __name__ == "__main__":
    # Travel-specific test texts
    test_texts = [
        "I want to book a flight from New York to London on December 25th",
        "Looking for a 5-star hotel near the Eiffel Tower in Paris",
        "Need a taxi from JFK Airport to Manhattan",
        "What's the best time to visit Mount Fuji?",
        "Book a table at The French Laundry restaurant in Napa Valley"
    ]
    
    # Initialize benchmark
    benchmark = FastNERBenchmark(
        model_path="/Users/skamalad/Library/Mobile Documents/com~apple~CloudDocs/All Tutorials/Programming/C/NER Extraction/model/weights.npz",
        tokenizer_name="bert-base-cased"
    )
    
    # Display NER predictions for each text
    print("\nNER Predictions:")
    print("=" * 50)
    for text in test_texts:
        print(f"\nText: {text}")
        tokens = benchmark.preprocess_text(text)
        tags, _ = benchmark.run_inference(tokens)  # Note: run_inference returns a tuple
        
        # Convert numeric tags back to labels
        tag_names = ['O', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-TRANSPORT', 'I-TRANSPORT', 
                    'B-ACCOMMODATION', 'I-ACCOMMODATION']
        
        # Group entities
        current_entity = None
        current_text = []
        entities = []
        
        token_texts = benchmark.tokenizer.convert_ids_to_tokens(tokens)
        
        for token, tag_id in zip(token_texts, tags):
            # Convert tag_id to integer if it's numpy array
            if hasattr(tag_id, 'item'):
                tag_id = tag_id.item()
            tag = tag_names[tag_id]
            
            if tag == 'O':
                if current_entity:
                    entities.append((current_entity, ' '.join(current_text)))
                    current_entity = None
                    current_text = []
            elif tag.startswith('B-'):
                if current_entity:
                    entities.append((current_entity, ' '.join(current_text)))
                current_entity = tag[2:]
                current_text = [token]
            elif tag.startswith('I-'):
                if current_entity == tag[2:]:
                    current_text.append(token)
        
        if current_entity:
            entities.append((current_entity, ' '.join(current_text)))
        
        print("Entities found:")
        for entity_type, text in entities:
            print(f"- {entity_type}: {text}")
        print()
    
    # Run performance benchmark
    results = benchmark.benchmark_comparison(test_texts)
    
    print("\nBenchmark Results:")
    print("-" * 50)
    print("\nC Implementation:")
    print(f"Mean Latency: {results['c_implementation']['mean_latency']:.2f}ms")
    print(f"P95 Latency: {results['c_implementation']['p95_latency']:.2f}ms")
    print(f"Throughput: {results['c_implementation']['throughput']:.2f} tokens/sec")
    print(f"Memory Usage: {results['c_implementation']['memory']:.2f}MB")
    
    if 'transformer' in results:
        print("\nTransformer Model:")
        print(f"Mean Latency: {results['transformer']['mean_latency']:.2f}ms")
        print(f"P95 Latency: {results['transformer']['p95_latency']:.2f}ms")
        print(f"Throughput: {results['transformer']['throughput']:.2f} tokens/sec")
        print(f"Memory Usage: {results['transformer']['memory']:.2f}MB")