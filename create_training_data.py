import numpy as np
from src.vocabulary import Vocabulary
from src.data_generator import create_training_data
import os

def main():
    print("Creating training data...")
    
    # Create output directory
    os.makedirs("model", exist_ok=True)
    
    # Generate synthetic data
    num_samples = 5000
    sentences, bio_tags = create_training_data(num_samples)
    
    print(f"\nGenerated {len(sentences)} sentences")
    print("\nExample sentences:")
    for i in range(5):
        print(f"{i+1}. Text: {sentences[i]}")
        print(f"   Tags: {bio_tags[i]}\n")
    
    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(sentences, min_freq=2)
    vocab.save("model/vocab.npz")
    
    # Create tag mapping
    tag_names = sorted(list({tag for tags in bio_tags for tag in tags}))
    tag2id = {tag: i for i, tag in enumerate(tag_names)}
    id2tag = {i: tag for tag, i in tag2id.items()}
    
    print(f"\nTag names: {tag_names}")
    
    # Find maximum sequence length
    max_length = max(len(sent.split()) for sent in sentences)
    print(f"\nMaximum sequence length: {max_length}")
    
    # Convert sentences and tags to arrays with padding
    token_ids = np.array([
        vocab.encode(sent, max_length=max_length)
        for sent in sentences
    ])
    
    tag_ids = np.zeros((len(bio_tags), max_length), dtype=np.int32)
    for i, tags in enumerate(bio_tags):
        # Convert tags to IDs
        tag_seq = [tag2id[tag] for tag in tags]
        # Pad with O tag
        tag_seq.extend([tag2id['O']] * (max_length - len(tag_seq)))
        tag_ids[i, :] = tag_seq
    
    print(f"\nToken IDs shape: {token_ids.shape}")
    print(f"Tag IDs shape: {tag_ids.shape}")
    
    # Calculate transition and emission probabilities
    num_tags = len(tag_names)
    num_words = len(vocab.word2id)
    
    # Initialize matrices with small random values
    transitions = np.ones((num_tags, num_tags))  # Start with 1 for Laplace smoothing
    emissions = np.ones((num_words, num_tags))   # Start with 1 for Laplace smoothing
    
    # Count transitions
    for tags in bio_tags:
        tag_ids = [tag2id[tag] for tag in tags]
        for i in range(len(tag_ids)-1):
            transitions[tag_ids[i], tag_ids[i+1]] += 1
    
    # Count emissions
    for sent, tags in zip(sentences, bio_tags):
        words = vocab.encode(sent)
        tag_ids = [tag2id[tag] for tag in tags]
        for word_id, tag_id in zip(words, tag_ids):
            emissions[word_id, tag_id] += 1
    
    # Convert to probabilities
    transitions = transitions / transitions.sum(axis=1, keepdims=True)
    emissions = emissions / emissions.sum(axis=1, keepdims=True)
    
    # Convert to log probabilities
    transitions = np.log(transitions)
    emissions = np.log(emissions)
    
    print("\nTransition matrix:")
    print(transitions)
    print("\nEmission matrix shape:", emissions.shape)
    
    # Save weights
    np.savez_compressed(
        "model/weights.npz",
        transitions=transitions.astype(np.float32),
        emissions=emissions.astype(np.float32),
        tag_names=np.array(tag_names, dtype=str)
    )
    print("\nSaved model weights and mappings to model/weights.npz")

if __name__ == "__main__":
    main()
