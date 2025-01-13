import ctypes
import numpy as np
from pathlib import Path
import platform
from .vocabulary import Vocabulary

class FastNER:
    def __init__(self, model_path: str):
        print(f"Python: Loading model from {model_path}")
        
        # Load vocabulary
        vocab_path = str(Path(model_path).parent / "vocab.npz")
        self.vocab = Vocabulary.load(vocab_path)
        print("Python: Loaded vocabulary")
        
        # Load the shared library
        lib_files = list(Path('.').glob('libner*.dylib')) + list(Path('.').glob('libner*.so'))
        if not lib_files:
            lib_files = list(Path('build/lib*').glob('libner*.dylib')) + list(Path('build/lib*').glob('libner*.so'))
        
        if not lib_files:
            raise RuntimeError("Could not find libner shared library")
            
        lib_path = lib_files[0]
        print(f"Python: Loading library from {lib_path}")
        self.lib = ctypes.CDLL(str(lib_path))
        print("Python: Library loaded successfully")
        
        # Define argument and return types
        self.lib.init_model.argtypes = [
            ctypes.c_int32,  # state_size
            ctypes.c_int32,  # vocab_size
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # transitions
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')   # emissions
        ]
        self.lib.init_model.restype = ctypes.c_void_p
        
        self.lib.decode_sequence.argtypes = [
            ctypes.c_void_p,  # model_ptr
            np.ctypeslib.ndpointer(dtype=np.int32),  # tokens
            ctypes.c_int32,  # seq_length
            np.ctypeslib.ndpointer(dtype=np.int32),  # output_tags
            np.ctypeslib.ndpointer(dtype=np.float32)  # output_scores
        ]
        self.lib.decode_sequence.restype = None
        
        self.lib.free_model.argtypes = [ctypes.c_void_p]
        self.lib.free_model.restype = None
        
        print("Python: Function signatures defined")
        
        # Load model weights
        print("Python: Loading weights file...")
        weights = np.load(model_path)
        print(f"Python: Available arrays in weights: {list(weights.keys())}")
        
        # Ensure arrays are contiguous and float32
        self.transitions = np.ascontiguousarray(weights['transitions'].astype(np.float32))
        self.emissions = np.ascontiguousarray(weights['emissions'].astype(np.float32))
        
        # Create tag mappings
        tag_names = weights['tag_names']
        self.tag2id = {tag: i for i, tag in enumerate(tag_names)}
        self.id2tag = {i: tag for i, tag in enumerate(tag_names)}
        
        print(f"Python: Transitions shape: {self.transitions.shape}, strides: {self.transitions.strides}")
        print(f"Python: Emissions shape: {self.emissions.shape}, strides: {self.emissions.strides}")
        
        self.state_size = self.transitions.shape[0]
        self.vocab_size = self.emissions.shape[0]
        print(f"Python: state_size={self.state_size}, vocab_size={self.vocab_size}")
        
        # Initialize model with correct argument order
        print("Python: Calling init_model...")
        self.model_ptr = self.lib.init_model(
            self.state_size,
            self.vocab_size,
            self.transitions,
            self.emissions
        )
        print(f"Python: init_model returned {self.model_ptr}")
        
        if not self.model_ptr:
            raise RuntimeError("Failed to initialize NER model")
        print("Python: Model initialized successfully")
    
    def preprocess_text(self, text: str) -> np.ndarray:
        """Convert text to token IDs using vocabulary."""
        return self.vocab.encode(text)
    
    def __call__(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform NER inference on input tokens.
        
        Args:
            tokens: Input token IDs as int32 numpy array
            
        Returns:
            Tuple of (predicted tags, tag scores)
        """
        tokens = np.ascontiguousarray(tokens, dtype=np.int32)
        seq_length = len(tokens)
        output_tags = np.zeros(seq_length, dtype=np.int32)
        output_scores = np.zeros(seq_length, dtype=np.float32)
        
        print(f"Python: Running inference on sequence of length {seq_length}")
        print(f"Python: Input tokens: {tokens}")
        
        self.lib.decode_sequence(
            self.model_ptr,
            tokens,
            seq_length,
            output_tags,
            output_scores
        )
        
        print(f"Python: Inference complete")
        print(f"Python: Predicted tags: {output_tags}")
        print(f"Python: Tag scores: {output_scores}")
        
        return output_tags, output_scores
    
    def format_results(self, tokens: list[str], tags: np.ndarray, scores: np.ndarray) -> list[tuple[str, str, float]]:
        """
        Format the results into a list of (token, tag, score) tuples.
        
        Args:
            tokens: List of input tokens
            tags: Predicted tag IDs
            scores: Tag scores
            
        Returns:
            List of (token, tag_name, score) tuples
        """
        results = []
        for token, tag_id, score in zip(tokens, tags, scores):
            tag_name = self.id2tag.get(int(tag_id), f"Unknown-{tag_id}")
            results.append((token, tag_name, score))
        return results
    
    def __del__(self):
        if hasattr(self, 'model_ptr') and self.model_ptr:
            print("Python: Freeing model memory")
            self.lib.free_model(self.model_ptr)
            print("Python: Model freed")

# Example usage
if __name__ == "__main__":
    import time
    
    # Create model
    ner = FastNER("path/to/model/weights.npz")
    
    # Example input
    text = "This is an example sentence"
    tokens = ner.preprocess_text(text)
    
    # Measure inference time
    start = time.perf_counter()
    tags, scores = ner(tokens)
    end = time.perf_counter()
    
    print(f"Inference time: {(end-start)*1000:.2f}ms")
    print(f"Predicted tags: {tags}")
    print(f"Tag scores: {scores}")
    
    # Format results
    formatted_results = ner.format_results(text.split(), tags, scores)
    print("Formatted Results:")
    for token, tag, score in formatted_results:
        print(f"{token}: {tag} (score={score:.4f})")