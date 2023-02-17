import numpy as np

# Global variables and some utility functions

window_width = 1400
window_height = 900

saved_weights_path = ("saved_weights")


def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results
