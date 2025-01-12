import ctypes
import numpy as np
from pathlib import Path

class FastNER:
    def __init__(self, model_path: str):
        # Load the shared library
        lib_path = Path(__file__).parent / "libner.so"
        self.lib = ctypes.CDLL(str(lib_path))
        
        # Define argument and return types
        self.lib.init_model.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_int,
            ctypes.c_int
        ]
        self.lib.init_model.restype = ctypes.c_void_p
        
        self.lib.decode_sequence.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.int32),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.float32)
        ]
        
        # Load model weights
        weights = np.load(model_path)
        self.transitions = weights['transitions'].astype(np.float32)
        self.emissions = weights['emissions'].astype(np.float32)
        self.state_size = self.transitions.shape[0]
        self.vocab_size = self.emissions.shape[0]
        
        # Initialize model
        self.model_ptr = self.lib.init_model(
            self.transitions,
            self.emissions,
            self.state_size,
            self.vocab_size
        )
    
    def __call__(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform NER inference on input tokens.
        
        Args:
            tokens: Input token IDs as int32 numpy array
            
        Returns:
            Tuple of (predicted tags, tag scores)
        """
        seq_length = len(tokens)
        output_tags = np.zeros(seq_length, dtype=np.int32)
        output_scores = np.zeros(seq_length, dtype=np.float32)
        
        self.lib.decode_sequence(
            self.model_ptr,
            tokens.astype(np.int32),
            seq_length,
            output_tags,
            output_scores
        )
        
        return output_tags, output_scores
    
    def __del__(self):
        if hasattr(self, 'model_ptr'):
            self.lib.free_model(self.model_ptr)

# Example usage
if __name__ == "__main__":
    import time
    
    # Create model
    ner = FastNER("path/to/model/weights.npz")
    
    # Example input
    tokens = np.array([1, 4, 2, 5, 3], dtype=np.int32)
    
    # Measure inference time
    start = time.perf_counter()
    tags, scores = ner(tokens)
    end = time.perf_counter()
    
    print(f"Inference time: {(end-start)*1000:.2f}ms")
    print(f"Predicted tags: {tags}")
    print(f"Tag scores: {scores}")