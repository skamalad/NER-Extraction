import numpy as np
from typing import List, Tuple
import time

class PythonNER:
    def __init__(self, state_size: int, vocab_size: int, transitions: np.ndarray, emissions: np.ndarray):
        """Initialize the NER model with transition and emission matrices.
        
        Args:
            state_size: Number of possible tags/states
            vocab_size: Size of the vocabulary
            transitions: Log probability matrix [state_size x state_size]
            emissions: Log probability matrix [vocab_size x state_size]
        """
        self.state_size = state_size
        self.vocab_size = vocab_size
        self.transitions = transitions
        self.emissions = emissions
        
    def viterbi_decode(self, tokens: List[int]) -> List[int]:
        """Decode a sequence of tokens into their most likely tags using Viterbi algorithm.
        
        Args:
            tokens: List of token IDs to decode
            
        Returns:
            List of predicted tag IDs
        """
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

def benchmark_python_ner(model: PythonNER, tokens: List[List[int]], num_runs: int = 100) -> Tuple[float, float, float]:
    """Benchmark the Python NER implementation.
    
    Args:
        model: PythonNER model instance
        tokens: List of token sequences to process
        num_runs: Number of benchmark iterations
        
    Returns:
        Tuple of (mean_latency, p95_latency, throughput)
    """
    latencies = []
    total_tokens = sum(len(seq) for seq in tokens)
    
    for _ in range(num_runs):
        start_time = time.time()
        for seq in tokens:
            model.viterbi_decode(seq)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    throughput = (total_tokens * num_runs) / sum(latencies) * 1000  # tokens/sec
    
    return mean_latency, p95_latency, throughput

if __name__ == "__main__":
    # Load weights
    weights = np.load('model/weights.npz')
    transitions = weights['transitions']
    emissions = weights['emissions']
    
    # Create model
    model = PythonNER(
        state_size=transitions.shape[0],
        vocab_size=emissions.shape[0],
        transitions=transitions,
        emissions=emissions
    )
    
    # Example usage
    tokens = [101, 1037, 2215, 2000, 2338, 1037, 3246, 2013, 2137, 2259, 2000, 2628, 102]
    tags = model.viterbi_decode(tokens)
    print(f"Tokens: {tokens}")
    print(f"Predicted tags: {tags}")
