import numpy as np
import os
import sys

# Ensure simulator can be imported 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.simulator import WiFiSimulator

def generate_dataset(num_samples=1000):
    sim = WiFiSimulator()
    X = []
    y = []
    print(f"Generating {num_samples} simulation samples...")
    for i in range(num_samples):
        if i > 0 and i % 500 == 0:
            print(f"Generated {i} samples...")
        res = sim.simulate()
        X.append(res['features'])
        y.append(res['label_idx'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Feature Scaling (Min-Max Normalization)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    
    diff = X_max - X_min
    diff[diff == 0] = 1e-6
    X_norm = (X - X_min) / diff
    
    return X_norm, y, X_min, diff
