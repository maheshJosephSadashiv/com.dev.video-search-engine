import numpy as np


def normalize_np_array(np_array, mini, maxi):
    for i, lis in enumerate(np_array):
        lis = np.array(lis)
        np_array[i] = (lis - mini) / (maxi - mini)
    return np_array
