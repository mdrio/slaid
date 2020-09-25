import pickle
import numpy as np


class Model:
    def __init__(self, filename: str):
        with open(filename, 'rb') as f:
            self._model = pickle.load(f)

    def predict(self, array: np.array) -> np.array:
        return self._model.predict(array)


class RandomModel:
    def predict(self, array):
        return np.random.uniform(0, 1, array.shape[0])
