from src.fast_ner import FastNER

ner = FastNER("model/weights.npz")

text="Apple Inc. is planning to open a new store in New York City next month."
tokens = ner.preprocess_text(text)
tags, scores = ner.run_inference(tokens)

print(f"Predicted tags: {tags}")
print(f"Tag scores: {scores}")

