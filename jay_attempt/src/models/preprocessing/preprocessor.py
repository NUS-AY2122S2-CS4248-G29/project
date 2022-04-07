import numpy as np

class Preprocessor:
    def __init__(self) -> None:
        pass

    def get_vocabulary_size(self) -> int:
        pass

    def adapt(self, raw_text_arr: np.ndarray) -> None:
        pass

    def preprocess(self, raw_text_arr: np.ndarray):
        pass

    def __call__(self, raw_text_arr: np.ndarray):
        return self.preprocess(raw_text_arr)
