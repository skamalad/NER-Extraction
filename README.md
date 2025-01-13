# NER Extraction

A fast and efficient Named Entity Recognition (NER) implementation in C with Python bindings. This project implements a Hidden Markov Model (HMM) based approach for Named Entity Recognition, with the core algorithm written in C for performance and Python bindings for ease of use.

## Features

- Core NER implementation in C for maximum performance
- Python bindings using ctypes for easy integration
- HMM-based sequence labeling
- Fast Viterbi decoding algorithm
- Support for custom models and weights
- CMake-based build system

## Project Structure

```
.
├── src/
│   ├── ner.c         # Core C implementation
│   ├── ner.h         # C header file
│   └── fast_ner.py   # Python bindings
├── test_ner.py       # Test suite
├── CMakeLists.txt    # CMake build configuration
└── model/            # Model weights
    └── weights.npz   # Numpy weights file
```

## Building

1. Create build directory:
```bash
mkdir build
cd build
```

2. Configure and build:
```bash
cmake ..
make
```

3. Copy library to project root:
```bash
cp libner.dylib ..
```

## Usage

```python
from src.fast_ner import FastNER

# Initialize the NER model
ner = FastNER("model/weights.npz")

# Process some text
text = "I want to fly from New York to Paris on Air France"
tokens = text.split()

# Get token IDs
token_ids = ner.preprocess_text(text)

# Run inference
tags, scores = ner(token_ids)

# Format results
results = ner.format_results(tokens, tags, scores)
for token, tag, score in results:
    print(f"{token:12} {tag:8} {score:8.4f}")
```

## Building from Source

Requirements:
- C compiler (gcc/clang)
- CMake 3.x
- Python 3.x
- NumPy

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
