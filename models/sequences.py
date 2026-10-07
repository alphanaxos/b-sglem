import numpy as np

def make_sequences(X_arr, y_arr, time_steps=24):
    """
    Creates overlapping sequences from array-like inputs for time-series modeling.

    Args:
        X_arr (np.ndarray): Scaled feature matrix.
        y_arr (np.ndarray): Scaled target vector.
        time_steps (int): Sequence window length. Defaults to 24.

    Returns:
        tuple: (X_sequences, y_sequences) as numpy arrays.
    """
    Xs, ys = [], []
    for i in range(len(X_arr) - time_steps):
        Xs.append(X_arr[i:i+time_steps])
        ys.append(y_arr[i+time_steps])
    return np.array(Xs), np.array(ys)
