import numpy as np

def create_sliding_windows(data: np.ndarray, window_size: int, stride: int = 1):
    """
    Convert time-series data into overlapping sliding windows.
    """

    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        windows.append(window.flatten())

    return np.array(windows)
