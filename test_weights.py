import numpy as np

print("Loading weights file...")
weights = np.load('model/weights.npz')
print(f"Available arrays: {list(weights.keys())}")

transitions = weights['transitions']
emissions = weights['emissions']

print(f"\nTransitions shape: {transitions.shape}")
print(f"Transitions dtype: {transitions.dtype}")
print(f"Transitions min/max: {transitions.min():.2f}/{transitions.max():.2f}")

print(f"\nEmissions shape: {emissions.shape}")
print(f"Emissions dtype: {emissions.dtype}")
print(f"Emissions min/max: {emissions.min():.2f}/{emissions.max():.2f}")
