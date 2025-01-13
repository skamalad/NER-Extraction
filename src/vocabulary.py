import numpy as np
from collections import defaultdict
from typing import List, Dict, Set
import re

class Vocabulary:
    def __init__(self):
        # Special tokens
        self.PAD = "<pad>"   # Padding token
        self.UNK = "<unk>"   # Unknown token
        self.BOS = "<bos>"   # Beginning of sequence
        self.EOS = "<eos>"   # End of sequence
        
        # Initialize word-to-id and id-to-word mappings
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.word_freq: Dict[str, int] = defaultdict(int)
        
        # Add special tokens
        for token in [self.PAD, self.UNK, self.BOS, self.EOS]:
            self._add_word(token)
    
    def _add_word(self, word: str) -> int:
        """Add a word to the vocabulary and return its ID."""
        if word not in self.word2id:
            word_id = len(self.word2id)
            self.word2id[word] = word_id
            self.id2word[word_id] = word
        return self.word2id[word]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text by lowercasing and normalizing."""
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r'[^a-z0-9\s\']', ' ', text)
        
        return text.strip()
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """Build vocabulary from a list of texts."""
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Count word frequencies
        for text in texts:
            text = self._preprocess_text(text)
            for word in text.split():
                self.word_freq[word] += 1
        
        # Add words that meet minimum frequency
        for word, freq in self.word_freq.items():
            if freq >= min_freq:
                self._add_word(word)
        
        print(f"Vocabulary size: {len(self.word2id)}")
    
    def encode(self, text: str, max_length: int = None) -> np.ndarray:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length. If None, use actual length.
            
        Returns:
            Token IDs as int32 numpy array
        """
        text = self._preprocess_text(text)
        tokens = text.split()
        
        # Convert tokens to IDs
        token_ids = [
            self.word2id.get(token, self.word2id[self.UNK])
            for token in tokens
        ]
        
        if max_length is not None:
            # Truncate if too long
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            # Pad if too short
            elif len(token_ids) < max_length:
                token_ids.extend([self.word2id[self.PAD]] * (max_length - len(token_ids)))
        
        return np.array(token_ids, dtype=np.int32)
    
    def decode(self, token_ids: np.ndarray) -> List[str]:
        """Convert token IDs back to words."""
        words = []
        for id_ in token_ids:
            word = self.id2word.get(int(id_), self.UNK)
            if word == self.PAD:
                break
            words.append(word)
        return words
    
    def save(self, path: str):
        """Save vocabulary to file."""
        data = {
            'word2id': self.word2id,
            'id2word': self.id2word,
            'word_freq': dict(self.word_freq)
        }
        np.savez_compressed(path, **data)
        print(f"Saved vocabulary to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from file."""
        vocab = cls()
        data = np.load(path, allow_pickle=True)
        vocab.word2id = data['word2id'].item()
        vocab.id2word = data['id2word'].item()
        vocab.word_freq = defaultdict(int, data['word_freq'].item())
        print(f"Loaded vocabulary with {len(vocab.word2id)} words")
        return vocab
