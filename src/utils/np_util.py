import numpy as np


def append_zeros(array, size):
    current_size = array.size
    if current_size < size:
        zeros_to_add = size - current_size
        zeros_array = np.zeros(zeros_to_add)
        return np.concatenate((array, zeros_array))
    else:
        return array
