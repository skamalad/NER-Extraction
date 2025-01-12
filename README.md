# NER Extraction

A fast and efficient Named Entity Recognition (NER) implementation in C with Python bindings. This project implements a Hidden Markov Model (HMM) based approach for Named Entity Recognition, with the core algorithm written in C for performance and Python bindings for ease of use.

## Features

- Core NER implementation in C for maximum performance
- Python bindings for easy integration
- HMM-based sequence labeling
- Fast Viterbi decoding algorithm
- Support for custom models and weights

## Project Structure

```
.
├── src/
│   ├── ner.c         # Core C implementation
│   ├── ner.h         # C header file
│   └── fast_ner.py   # Python bindings
├── test_ner.py       # Test suite
├── create_weights.py # Weight initialization
└── python_ner.py    # Python implementation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/skamalad/NER-Extraction.git
cd NER-Extraction
```

2. Build the C extension:
```bash
python setup.py build_ext --inplace
```

## Usage

```python
from src.fast_ner import NER

# Initialize the NER model with your weights
ner = NER(transitions, emissions)

# Decode a sequence
tags, scores = ner.decode_sequence(tokens)
```

## Building from Source

Requirements:
- C compiler (gcc/clang)
- Python 3.x
- NumPy

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
