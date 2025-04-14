import pickle
import os
import numpy as np
from utils import NTupleApproximator
# import json

def save_weights(approximator, filename):
    weights = []
    for i in range(len(approximator.weights)):
        weights.append(approximator.weights[i])
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Weights saved to {filename}")

def load_weights(filename):
    with open(filename, 'rb') as f:
        weights = pickle.load(f)   
    return weights

def main():
    file_name = "./model/value_revised_approximator_11200.pkl"
    # Assume: self.weights = [defaultdict(float), defaultdict(float), ...]
    # Convert to list of dicts
    # Load the approximator
    with open(file_name, 'rb') as f:
        approximator = pickle.load(f)
    
    for w in approximator.weights:
       for k, v in w.items():
           if isinstance(v, np.float64):
               w[k] = float(v)
    
    with open("value_revised_approximator_11200.pkl", 'wb') as f:
        pickle.dump(approximator, f)



if __name__ == "__main__":
    main()